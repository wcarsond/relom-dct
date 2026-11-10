import urllib.request, json, struct, ctypes, numpy as np, time, os, sys
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
dll_path = os.environ.get("GGML_BASE_DLL", "ggml-base.dll")
base = ctypes.CDLL(dll_path)
base.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
base.ggml_quantize_chunk.restype = ctypes.c_size_t
GGML_TYPE_Q12H = 43

idx = json.loads(urllib.request.urlopen("https://huggingface.co/openai/gpt-oss-120b/resolve/main/model.safetensors.index.json", timeout=15).read().decode())
wm = idx["weight_map"]

def gguf_to_hf(gname):
    if gname == "token_embd.weight": return "model.embed_tokens.weight"
    if gname == "output.weight": return "lm_head.weight"
    p = gname.split(".")
    l, sub = p[1], p[2]
    m = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj"}[sub]
    return f"model.layers.{l}.{m}.weight"

# 1. Parse source GGUF
src_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SRC_GGUF", "gpt-oss-120b-MXFP4.gguf")
f_src = open(src_path, "rb")
magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f_src.read(24))
def read_str(): return f_src.read(struct.unpack("<Q", f_src.read(8))[0])
def skip_val(vt):
    if vt in (0,1,7): f_src.read(1)
    elif vt in (2,3): f_src.read(2)
    elif vt in (4,5,6): f_src.read(4)
    elif vt in (10,11,12): f_src.read(8)
    elif vt == 8: read_str()
    elif vt == 9:
        at, al = struct.unpack("<IQ", f_src.read(12))
        for _ in range(al): skip_val(at)
for _ in range(n_kv):
    read_str()
    skip_val(struct.unpack("<I", f_src.read(4))[0])
kv_end = f_src.tell()

tensors = []
for _ in range(n_tensors):
    name = read_str().decode("utf-8")
    nd = struct.unpack("<I", f_src.read(4))[0]
    dims = struct.unpack(f"<{nd}Q", f_src.read(8 * nd))
    ttype, offset = struct.unpack("<IQ", f_src.read(12))
    tensors.append({"name": name, "dims": dims, "ttype": ttype, "offset": offset})
hdr_end = f_src.tell()
data_start = (hdr_end + 31) // 32 * 32
f_src.seek(0, 2)
file_size = f_src.tell()
by_off = sorted(tensors, key=lambda x: x["offset"])
for i, t in enumerate(by_off):
    next_off = by_off[i+1]["offset"] if i+1 < len(by_off) else (file_size - data_start)
    t["n_bytes"] = next_off - t["offset"]

# 2. Fetch shard headers in parallel
q8_tensors = [t for t in tensors if t["ttype"] == 8]
shards = sorted(set(wm[gguf_to_hf(t["name"])] for t in q8_tensors))
shard_headers = {}
def fetch_hdr(sh):
    url = f"https://huggingface.co/openai/gpt-oss-120b/resolve/main/{sh}"
    hl = struct.unpack("<Q", urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}), timeout=20).read())[0]
    hdr = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{8+hl-1}"}), timeout=20).read().decode())
    return sh, url, 8 + hl, hdr

with ThreadPoolExecutor(max_workers=8) as ex:
    for sh, url, base_off, hdr in ex.map(fetch_hdr, shards):
        shard_headers[sh] = (url, base_off, hdr)
print(f"Fetched all {len(shards)} shard headers in {time.time()-t0:.1f}s!")

# 3. Download ONLY the 146 BF16 backbone tensors via HTTP Range and quantize to Q12H in parallel!
q12h_blobs = {}
def fetch_and_quantize(t):
    gname = t["name"]
    hname = gguf_to_hf(gname)
    sh = wm[hname]
    url, base_off, hdr = shard_headers[sh]
    s, e = hdr[hname]["data_offsets"]
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={base_off+s}-{base_off+e-1}"})
            raw = urllib.request.urlopen(req, timeout=120).read()
            if len(raw) == (e - s):
                break
        except Exception as err:
            if attempt == 3: raise
            time.sleep(1)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    n_per_row = int(t["dims"][0])
    nrows = int(t["dims"][1])
    # Exact Q12H byte size: nrows * (n_per_row // 32) * 49? Let's allocate 2x and slice to returned sz!
    dst = np.zeros((nrows * n_per_row * 2), dtype=np.uint8)
    sz = base.ggml_quantize_chunk(GGML_TYPE_Q12H, f32.ctypes.data, dst.ctypes.data, 0, nrows, n_per_row, None)
    return gname, dst[:sz].tobytes()

t1 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for idx_done, (gname, blob) in enumerate(ex.map(fetch_and_quantize, q8_tensors), 1):
        q12h_blobs[gname] = blob
        if idx_done % 25 == 0 or idx_done == len(q8_tensors):
            print(f"  [{idx_done}/{len(q8_tensors)}] quantized to Q12H ({time.time()-t1:.1f}s elapsed)")

# 4. Splice output GGUF
out_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT_GGUF", "gpt-oss-120b-ql-backbone-mxfp4.gguf")
t2 = time.time()
with open(out_path, "wb") as out:
    f_src.seek(0)
    out.write(f_src.read(kv_end))
    cur_off = 0
    plan = []
    for t in tensors:
        gname = t["name"]
        nb = gname.encode("utf-8")
        out.write(struct.pack("<Q", len(nb)) + nb)
        out.write(struct.pack("<I", len(t["dims"])))
        out.write(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
        if gname in q12h_blobs:
            blob = q12h_blobs[gname]
            raw_len = len(blob)
            aligned_len = (raw_len + 31) // 32 * 32
            out.write(struct.pack("<IQ", GGML_TYPE_Q12H, cur_off))
            plan.append(("MEM", blob, raw_len, aligned_len - raw_len))
        else:
            raw_len = t["n_bytes"]
            aligned_len = (raw_len + 31) // 32 * 32
            out.write(struct.pack("<IQ", t["ttype"], cur_off))
            plan.append(("FILE", data_start + t["offset"], raw_len, aligned_len - raw_len))
        cur_off += aligned_len
    pos = out.tell()
    pad = (32 - (pos % 32)) % 32
    if pad: out.write(b"\x00" * pad)
    for kind, src_ref, raw_len, pad_len in plan:
        if kind == "MEM":
            out.write(src_ref)
        else:
            f_src.seek(src_ref)
            rem = raw_len
            while rem > 0:
                chunk = f_src.read(min(rem, 32 * 1024 * 1024))
                out.write(chunk)
                rem -= len(chunk)
        if pad_len:
            out.write(b"\x00" * pad_len)

f_src.close()
print(f"Done! Wrote {out_path} ({os.path.getsize(out_path)/(1024**3):.2f} GiB) in {time.time()-t0:.1f}s total (splice write took {time.time()-t2:.1f}s)!")
