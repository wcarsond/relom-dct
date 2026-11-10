# dct_nvme_race.py - race Single-NVMe (M:) + Plain LRU vs Dual-Channel NVMe (C: + M: 2:1) + Reddit Score + DCT
# on the real Qwen3-30B-A3B 3206-token routing trace (route-log.qwen30b.sqlite), using actual unbuffered Win32 ReadFile
# transfers of 2.32 MiB expert records on both NVMe controllers.
# Usage:  python dct_nvme_race.py [trace_sqlite] [stripe_c_path] [stripe_m_path]
import ctypes, os, pathlib, sqlite3, sys, tempfile, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
FILE_FLAG_NO_BUFFERING, FILE_FLAG_RANDOM_ACCESS = 0x20000000, 0x10000000


class OVERLAPPED(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(OVERLAPPED)]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]

here = pathlib.Path(__file__).parent
out_file = here / "dct_nvme_race.output.txt"
db_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROUTE_LOG_DB", str(here / "route-log.qwen30b.sqlite"))
db = sqlite3.connect(db_path)
E = db.execute("SELECT MAX(expert)+1 FROM route").fetchone()[0]
L = db.execute("SELECT MAX(layer)+1 FROM route").fetchone()[0]
K = db.execute("SELECT COUNT(*) FROM route WHERE prompt=0 AND token=0 AND layer=0").fetchone()[0]
tokens = {}
for pr, t, l, e in db.execute("SELECT prompt, token, layer, expert FROM route ORDER BY prompt, token, layer"):
    tokens.setdefault((pr, t), [[] for _ in range(L)])[l].append(e)
seq = [tokens[k] for k in sorted(tokens)][:256]

STRIDE = 2383872  # 2.32 MiB per expert (Qwen3-30B-A3B int4 record stride, 4KB aligned)
fc = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DCT_STRIPE_C", os.path.join(tempfile.gettempdir(), "dct-stripe-c.bin"))
fm = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("DCT_STRIPE_M", r"M:\q12h-gguf\qwen3coder30b-a3b-bf16.gguf")
SZ_C = 1024 * 1024 * 1024
if not os.path.exists(fc) or os.path.getsize(fc) < SZ_C:
    with open(fc, "wb") as f:
        buf = os.urandom(16 * 1024 * 1024)
        for _ in range(SZ_C // len(buf)):
            f.write(buf)

N_BLOCKS_C = SZ_C // STRIDE
N_BLOCKS_M = (2 * 1024 * 1024 * 1024) // STRIDE

hc_list = [k32.CreateFileW(fc, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, None) for _ in range(4)]
hm_list = [k32.CreateFileW(fm, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, None) for _ in range(4)]
bufs = [k32.VirtualAlloc(None, STRIDE, 0x3000, 0x04) for _ in range(8)]


def read_rec(handle, buf, blk_idx):
    off = blk_idx * STRIDE
    ov = OVERLAPPED()
    ov.Offset = off & 0xFFFFFFFF
    ov.OffsetHigh = off >> 32
    got = wintypes.DWORD(0)
    k32.ReadFile(handle, buf, STRIDE, ctypes.byref(got), ctypes.byref(ov))


for i in range(16):
    read_rec(hm_list[i % 4], bufs[i % 4], i)

lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


for cap in (22, 32, 48):
    lru = [OrderedDict() for _ in range(L)]
    t0 = time.perf_counter()
    misses_1 = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for layers in seq:
            for l in range(L):
                miss = []
                for e in layers[l]:
                    if e in lru[l]:
                        lru[l].move_to_end(e)
                    else:
                        misses_1 += 1
                        if len(lru[l]) >= cap:
                            lru[l].popitem(last=False)
                        lru[l][e] = 1
                        miss.append((l * E + e) % N_BLOCKS_M)
                if miss:
                    futs = [ex.submit(read_rec, hm_list[i % 4], bufs[i % 4], blk) for i, blk in enumerate(miss)]
                    for f in futs:
                        f.result()
    dt_1 = time.perf_counter() - t0

    res = [set() for _ in range(L)]
    freq = [[0.0] * E for _ in range(L)]
    last_t = [[-999] * E for _ in range(L)]
    t0 = time.perf_counter()
    misses_2 = 0
    with ThreadPoolExecutor(max_workers=4) as ex_c, ThreadPoolExecutor(max_workers=2) as ex_m:
        for tidx, layers in enumerate(seq):
            batch_c, batch_m = [], []
            for l in range(L):
                if tidx > 0 and tidx % 8 == 0:
                    for e in range(E):
                        freq[l][e] *= 0.88
                cur = layers[l]
                for e in cur:
                    freq[l][e] += 1.0
                    if e in res[l]:
                        last_t[l][e] = tidx
                    else:
                        misses_2 += 1
                        last_t[l][e] = tidx
                        if len(res[l]) >= cap:
                            victim = min((r for r in res[l] if r not in cur),
                                         key=lambda r: freq[l][r] + (2.5 if (tidx - last_t[l][r]) <= 8 else 0.0))
                            res[l].remove(victim)
                        res[l].add(e)
                        rec_id = l * E + e
                        if rec_id % 3 < 2:
                            batch_c.append(rec_id % N_BLOCKS_C)
                        else:
                            batch_m.append(rec_id % N_BLOCKS_M)
            futs = [ex_c.submit(read_rec, hc_list[i % 4], bufs[i % 4], blk) for i, blk in enumerate(batch_c)] + \
                   [ex_m.submit(read_rec, hm_list[i % 2], bufs[4 + (i % 2)], blk) for i, blk in enumerate(batch_m)]
            for f in futs:
                f.result()
    dt_2 = time.perf_counter() - t0
    say(f"Cap={cap:2d}/128 ({100*cap/E:4.1f}%):")
    say(f"  1. Single M: + LRU (per-layer):   {misses_1:5d} misses ({misses_1*STRIDE/1024**3:5.2f} GiB) | {dt_1*1000/len(seq):5.1f} ms/tok ({len(seq)/dt_1:5.1f} tok/s)")
    say(f"  2. Dual C:+M: + Reddit + DCT:     {misses_2:5d} misses ({misses_2*STRIDE/1024**3:5.2f} GiB) | {dt_2*1000/len(seq):5.1f} ms/tok ({len(seq)/dt_2:5.1f} tok/s) -> {dt_1/dt_2:.2f}x faster!")

for h in hc_list + hm_list:
    k32.CloseHandle(h)
for b in bufs:
    k32.VirtualFree(b, 0, 0x8000)
if os.path.exists(fc):
    os.remove(fc)
out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
