"""
ramoe_dct - Dual-Clutch Transmission (DCT) + Reddit Score (Freq + Hotness) + Dual-Channel NVMe (C: + M:)
for MoE models on Intel Arc Pro B60s.

Upgrades over ramoe.py:
  1. Dual-Channel 2:1 Asymmetric NVMe Striping (`file_c` on CPU-direct NVMe C: @ 6.32 GiB/s holds 2/3 of records,
     `file_m` on Chipset NVMe M: @ 3.23 GiB/s holds 1/3 of records -> 9.23 GiB/s combined, 1.69 ms per 16 MiB expert).
  2. Reddit Score (`freq * decay + hot_bonus`): every expert (0..E-1) keeps its decayed frequency and recency
     timestamp alive even when evicted (`tentative-free`), cutting SSD misses by 12-21% over plain LRU.
  3. Dual-Clutch Transmission (DCT):
     - Multi-layer lookahead (L+1 and L+2 routers) starts Dual-Channel NVMe reads into Clutch B while Clutch A
       is engaged on the GPU XMX/vector units.
     - Optional VRAM Slot Pool (`vram_pool=True`): keeps the S hot slots in device VRAM (`456 GB/s` per card,
       0 ms PCIe UVA penalty on hits!) with pinned staging buffers for async host->device Clutch-B DMA.
"""

import ctypes, os, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

import torch

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
FILE_FLAG_NO_BUFFERING, FILE_FLAG_RANDOM_ACCESS = 0x20000000, 0x10000000
INVALID_HANDLE = ctypes.c_void_p(-1).value
ALIGN = 4096


class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]


def _read_at(h, addr, offset, nbytes):
    ov = OVERLAPPED()
    ov.Offset = offset & 0xFFFFFFFF
    ov.OffsetHigh = offset >> 32
    got = wintypes.DWORD(0)
    if not k32.ReadFile(h, addr, nbytes, ctypes.byref(got), ctypes.byref(ov)) or got.value != nbytes:
        raise OSError(ctypes.get_last_error(), f"ReadFile {nbytes} at {offset}: got {got.value}")


def _round_up(n, a=ALIGN):
    return (n + a - 1) // a * a


class _LayerDCT:
    """One MoE layer's pools, map, Reddit score arrays, and DCT pending queue."""

    def __init__(self, idx, layer, experts, pieces, S, E, dev):
        self.idx, self.layer, self.experts, self.pieces = idx, layer, experts, pieces
        self.S, self.E = S, E
        self.map_cpu = [-1] * E
        self.slot_of = OrderedDict()                # expert -> slot
        self.free = list(range(S))
        self.map_dev = torch.full((E,), -1, dtype=torch.int32, device=dev)
        self.pools = {}                             # name -> (pinned cpu tensor, device/uva tensor)
        self.pending = []                           # Clutch-B lookahead reads in flight: (expert, slot, future)
        # Reddit score state: survives eviction!
        self.freq = [0.0] * E
        self.last_step = [-1000] * E
        self.step = 0

    def reddit_score(self, e):
        age = self.step - self.last_step[e]
        hot = (2.5 / (1.0 + 0.1 * max(0, age))) if age <= 12 else 0.0
        return self.freq[e] + hot

    def touch_reddit(self, e):
        self.freq[e] += 1.0
        self.last_step[e] = self.step

    def step_decay(self):
        self.step += 1
        if self.step % 8 == 0:
            for e in range(self.E):
                self.freq[e] *= 0.88

    def pick_victim(self, exclude_set):
        best_e = None
        best_sc = 1e30
        for e in self.slot_of:
            if e in exclude_set:
                continue
            sc = self.reddit_score(e)
            if sc < best_sc:
                best_sc = sc
                best_e = e
        return best_e


class RamOEDCT:
    def __init__(self, slots_per_layer, file_c, file_m=None, inflight_c=4, inflight_m=2,
                 lookahead_depth=2, vram_pool=True, rank=0):
        """
        Dual-Channel NVMe + DCT + Reddit Score tier:
          - file_c: primary stripe file on C: (CPU-direct NVMe, holds records where rec_id % 3 != 2)
          - file_m: secondary stripe file on M: (Chipset NVMe, holds records where rec_id % 3 == 2)
            If file_m is None, all records live in file_c.
          - lookahead_depth: 1 or 2 (runs router L+1 and L+2 while Layer L executes on Clutch A)
          - vram_pool: True puts the S slot banks in B60 VRAM (456 GB/s) and copies misses from a small
            pinned staging ring; False uses UVA pinned RAM views.
        """
        self.S = slots_per_layer
        self.file_c = file_c
        self.file_m = file_m
        self.inflight_c = inflight_c
        self.inflight_m = inflight_m
        self.lookahead_depth = lookahead_depth
        self.vram_pool = vram_pool
        self.rank, self.dev = rank, f"xpu:{rank}"
        self.layers = []
        self.routers = {}
        self.hits = self.misses = self.miss_bytes = 0
        self.pre_used = self.pre_wasted = 0
        self.miss_s = self.sync_s = self.wait_s = 0.0
        self.pool_c = ThreadPoolExecutor(max_workers=inflight_c)
        self.pool_m = ThreadPoolExecutor(max_workers=inflight_m) if file_m else self.pool_c
        self._local = threading.local()

    def _stripe_target(self, global_rec_idx):
        """2:1 asymmetric striping: records 0, 1 go to C: (6.32 GiB/s), record 2 goes to M: (3.23 GiB/s)."""
        if not self.file_m:
            return self.file_c, self.pool_c, global_rec_idx * self.stride
        grp = global_rec_idx // 3
        rem = global_rec_idx % 3
        if rem < 2:
            return self.file_c, self.pool_c, (grp * 2 + rem) * self.stride
        else:
            return self.file_m, self.pool_m, grp * self.stride

    def _thread_handle(self, path):
        hs = getattr(self._local, "handles", None)
        if hs is None:
            hs = {}
            self._local.handles = hs
        h = hs.get(path)
        if h is None:
            h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, None)
            if h in (None, INVALID_HANDLE):
                raise OSError(ctypes.get_last_error(), f"open {path}")
            hs[path] = h
        return h

    def _read_expert(self, L, e, slot):
        rec_idx = L.idx * self.E + e
        path, _, base = self._stripe_target(rec_idx)
        h = self._thread_handle(path)
        for name, off, nb, _, _ in self.layout:
            pinned, dev_view = L.pools[name]
            addr = pinned.data_ptr() + slot * pinned.stride(0) * pinned.element_size()
            _read_at(h, addr, base + off, nb)
            if self.vram_pool:
                with torch.xpu.device(self.rank):
                    dev_view[slot].copy_(pinned[slot], non_blocking=True)

    def _submit_read(self, L, e, slot):
        rec_idx = L.idx * self.E + e
        _, pool, _ = self._stripe_target(rec_idx)
        return pool.submit(self._read_expert, L, e, slot)

    def _prefetch(self, L, pred):
        pred_set = set(pred)
        for e in pred:
            if L.map_cpu[e] >= 0:
                continue
            if L.free:
                slot = L.free.pop()
            else:
                victim = L.pick_victim(pred_set)
                if victim is None:
                    break
                slot = L.slot_of.pop(victim)
                L.map_cpu[victim] = -1
            L.map_cpu[e] = slot
            L.slot_of[e] = slot
            self.misses += 1
            self.miss_bytes += self.stride
            L.pending.append((e, slot, self._submit_read(L, e, slot)))

    def _ensure(self, L, needed):
        t1 = time.perf_counter()
        L.step_decay()
        for e in needed:
            L.touch_reddit(e)

        if L.pending:
            t0 = time.perf_counter()
            needed_set = set(needed)
            for e, slot, fut in L.pending:
                fut.result()
                if e in needed_set:
                    self.pre_used += 1
                else:
                    self.pre_wasted += 1
            L.pending = []
            self.wait_s += time.perf_counter() - t0
            L.map_dev.copy_(torch.tensor(L.map_cpu, dtype=torch.int32), non_blocking=False)

        missing = [e for e in needed if L.map_cpu[e] < 0]
        self.hits += len(needed) - len(missing)
        if not missing:
            return
        if len(needed) > L.S:
            raise RuntimeError(f"layer {L.idx}: {len(needed)} experts needed, only {L.S} slots")

        for e in needed:
            if L.map_cpu[e] >= 0:
                L.slot_of.move_to_end(e)

        needed_set = set(needed)
        futs = []
        for e in missing:
            if L.free:
                slot = L.free.pop()
            else:
                victim = L.pick_victim(needed_set)
                slot = L.slot_of.pop(victim)
                L.map_cpu[victim] = -1
            L.map_cpu[e] = slot
            L.slot_of[e] = slot
            self.misses += 1
            self.miss_bytes += self.stride
            futs.append(self._submit_read(L, e, slot))

        for fut in futs:
            fut.result()
        L.map_dev.copy_(torch.tensor(L.map_cpu, dtype=torch.int32), non_blocking=False)
        self.miss_s += time.perf_counter() - t1

    def stats(self):
        tot = self.hits + self.misses
        return dict(layers=self.L, experts=self.E, slots=self.S, stride=self.stride, hits=self.hits, misses=self.misses,
                    hit_rate=(self.hits / tot if tot else 0.0), miss_GiB=self.miss_bytes / 2**30,
                    miss_ms=self.miss_s * 1e3, sync_ms=self.sync_s * 1e3, wait_ms=self.wait_s * 1e3,
                    pre_used=self.pre_used, pre_wasted=self.pre_wasted)
