r"""ql-roundtrip.input.py - is a "QL backbone" GGUF really bit-exact? For every Q12H tensor in the file, pull the
original bf16 tensor from Hugging Face (byte range, held in memory only), dequantize the GGUF's Q12H bytes with
the same ggml-base.dll that wrote them, and compare all 16 bits of every weight.

Usage:
    python -I ql-roundtrip.input.py 80b|120b
Output: ql-roundtrip.<which>.output.txt (one row per tensor that differs, then totals). Nothing is written to M:.

A weight that differs was "snapped" by patch 0005 (more than 4 outliers in its 256-weight block).
"""
import ctypes, json, os, struct, sys, time, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

which = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "80b":  dict(repo="Qwen/Qwen3-Coder-Next", gguf=r"M:\models-test\qwen3-coder-next\Qwen3-Coder-Next-QL-Backbone-Q4_K_XL.gguf"),
    "120b": dict(repo="openai/gpt-oss-120b",   gguf=r"M:\models-b60\gpt-oss-120b\gpt-oss-120b-QL-Backbone-MXFP4.gguf"),
}[which]
SUB = {  # same name map the surgical scripts used
    "ssm_out": "linear_attn.out_proj", "ffn_down_shexp": "mlp.shared_expert.down_proj",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
}
def gguf_to_hf(g):
    if g == "token_embd.weight": return "model.embed_tokens.weight"
    if g == "output.weight": return "lm_head.weight"
    p = g.split(".")
    return f"model.layers.{p[1]}.{SUB[p[2]]}.weight" if p[0] == "blk" and p[2] in SUB else None

Q12H, QK, BLOCK_BYTES = 43, 256, 412     # 412 bytes per 256 weights = 12.875 bpw

class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool),
                ("to_float", ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)),
                ("from_float_ref", ctypes.c_void_p)]
dll = ctypes.CDLL(os.environ.get("GGML_BASE_DLL", "ggml-base.dll"))
dll.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
tr = dll.ggml_get_type_traits(Q12H).contents
assert tr.type_name == b"q12h" and tr.blck_size == QK, (tr.type_name, tr.blck_size)
BLOCK_BYTES = tr.type_size
to_float = tr.to_float

out = open(os.path.join(HERE, f"ql-roundtrip.{which}.output.txt"), "a", encoding="utf-8")
def say(s):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + s
    print(line, flush=True); out.write(line + "\n"); out.flush()

# --- GGUF header ---
f = open(CFG["gguf"], "rb")
magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
def rstr(): return f.read(struct.unpack("<Q", f.read(8))[0])
def skip(vt):
    if vt in (0, 1, 7): f.read(1)
    elif vt in (2, 3): f.read(2)
    elif vt in (4, 5, 6): f.read(4)
    elif vt in (10, 11, 12): f.read(8)
    elif vt == 8: rstr()
    elif vt == 9:
        at, al = struct.unpack("<IQ", f.read(12))
        for _ in range(al): skip(at)
for _ in range(n_kv):
    rstr(); skip(struct.unpack("<I", f.read(4))[0])
tensors = []
for _ in range(n_tensors):
    name = rstr().decode()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
    tt, off = struct.unpack("<IQ", f.read(12))
    tensors.append(dict(name=name, dims=dims, tt=tt, off=off))
data_start = (f.tell() + 31) // 32 * 32
f.close()
ql = [t for t in tensors if t["tt"] == Q12H]
types = {}
for t in tensors: types[t["tt"]] = types.get(t["tt"], 0) + 1
say(f"=== {which}: {CFG['gguf']}")
say(f"tensors {len(tensors)}, by ggml type {dict(sorted(types.items()))}; Q12H tensors to check: {len(ql)}")

def get(url, rng=None, timeout=120):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if attempt == 4: raise
            time.sleep(2 * (attempt + 1))

root = f"https://huggingface.co/{CFG['repo']}/resolve/main/"
wm = json.loads(get(root + "model.safetensors.index.json"))["weight_map"]
hdrs = {}
for sh in sorted(set(wm[gguf_to_hf(t["name"])] for t in ql)):
    hl = struct.unpack("<Q", get(root + sh, "bytes=0-7"))[0]
    hdrs[sh] = (8 + hl, json.loads(get(root + sh, f"bytes=8-{8 + hl - 1}")))
say(f"{len(hdrs)} safetensors shard headers read")

def check(t):
    hname = gguf_to_hf(t["name"])
    sh = wm[hname]
    base, hdr = hdrs[sh]
    meta = hdr[hname]
    assert meta["dtype"] == "BF16", (hname, meta["dtype"])
    s, e = meta["data_offsets"]
    raw = get(root + sh, f"bytes={base + s}-{base + e - 1}")
    assert len(raw) == e - s
    want = np.frombuffer(raw, dtype=np.uint16)
    n = want.size
    assert n == int(np.prod(t["dims"])) and n % QK == 0, (t["name"], n, t["dims"])
    with open(CFG["gguf"], "rb") as g:
        g.seek(data_start + t["off"])
        q = g.read(n // QK * BLOCK_BYTES)
    got32 = np.empty(n, dtype=np.float32)
    to_float(ctypes.c_char_p(q), got32.ctypes.data, n)
    got = got32.view(np.uint32)
    bad = (got != (want.astype(np.uint32) << 16))
    nbad = int(bad.sum())
    worst, blocks = 0.0, 0
    if nbad:
        w32 = (want.astype(np.uint32) << 16).view(np.float32)
        blocks = int(bad.reshape(-1, QK).any(axis=1).sum())
        # size of each snapped weight against the largest weight in its own 256-weight block
        bmax = np.abs(w32).reshape(-1, QK).max(axis=1).repeat(QK)
        worst = float((np.abs(w32[bad] - got32[bad]) / bmax[bad]).max())
    return t["name"], n, nbad, blocks, worst

tot_n = tot_bad = tot_blocks = bad_tensors = 0
worst_all = 0.0
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (name, n, nbad, blocks, worst) in enumerate(ex.map(check, ql), 1):
        tot_n += n; tot_bad += nbad; tot_blocks += blocks; worst_all = max(worst_all, worst)
        if nbad:
            bad_tensors += 1
            say(f"  DIFF {name}: {nbad:,} of {n:,} weights ({100 * nbad / n:.5f}%) in {blocks:,} blocks; worst error = {worst:.3e} of block max")
        if i % 40 == 0 or i == len(ql):
            say(f"  [{i}/{len(ql)}] {tot_n / 1e6:,.0f}M weights checked, {tot_bad:,} differ, {time.time() - t0:.0f}s")
say(f"TOTAL {which}: {len(ql)} Q12H tensors, {tot_n:,} weights; {tot_bad:,} differ ({100 * tot_bad / max(tot_n, 1):.6f}%) "
    f"in {bad_tensors} tensors / {tot_blocks:,} blocks; worst error {worst_all:.3e} of its block's max")
say("VERDICT: " + ("BIT-EXACT, every weight" if tot_bad == 0 else "NOT bit-exact (weights were snapped)"))
