"""
npu_router_switchman.py - Offload MoE Router Gates across ALL 48 layers onto the
Intel(R) AI Boost NPU (and/or Arrow Lake iGPU) via OpenVINO so the discrete GPUs
(2x Arc Pro B60) and AVX2 CPU threads never pause to score lookahead tracks.

Usage:
    C:\\ai-tools\\yolo\\venv\\Scripts\\python.exe npu_router_switchman.py [path/to/moe.gguf]
"""
import os, struct, sys, time
import numpy as np
import openvino as ov
from openvino import opset13 as ops


def load_moe_gates_from_gguf(gguf_path):
    with open(gguf_path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        def read_str(): return f.read(struct.unpack("<Q", f.read(8))[0])
        def skip_val(vt):
            if vt in (0, 1, 7): f.read(1)
            elif vt in (2, 3): f.read(2)
            elif vt in (4, 5, 6): f.read(4)
            elif vt in (10, 11, 12): f.read(8)
            elif vt == 8: read_str()
            elif vt == 9:
                at, al = struct.unpack("<IQ", f.read(12))
                for _ in range(al): skip_val(at)
        for _ in range(n_kv):
            read_str()
            skip_val(struct.unpack("<I", f.read(4))[0])
        tmap = {}
        for _ in range(n_tensors):
            name = read_str().decode()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            ttype, off = struct.unpack("<IQ", f.read(12))
            tmap[name] = (dims, ttype, off)
        ds = (f.tell() + 31) // 32 * 32
        gate_keys = sorted([k for k in tmap if k.endswith(".ffn_gate_inp.weight")],
                           key=lambda x: int(x.split(".")[1]))
        n_layers = len(gate_keys)
        d_model, n_experts = tmap[gate_keys[0]][0]
        gates = np.zeros((n_layers, n_experts, d_model), dtype=np.float32)
        for l, k in enumerate(gate_keys):
            dims, ttype, off = tmap[k]
            f.seek(ds + off)
            gates[l] = np.frombuffer(f.read(int(n_experts * d_model * 4)), dtype=np.float32).reshape(n_experts, d_model)
    return gates


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "MOE_GGUF", "Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
    )
    t0 = time.time()
    gates = load_moe_gates_from_gguf(gguf_path)
    n_layers, n_experts, d_model = gates.shape
    print(f"Loaded {n_layers} layers x {n_experts} experts ({n_layers*n_experts:,} total experts, d={d_model}) in {time.time()-t0:.2f}s")

    W_flat = np.ascontiguousarray(gates.reshape(n_layers * n_experts, d_model).T, dtype=np.float32)
    param_x = ops.parameter([1, d_model], ov.Type.f32, name="x")
    const_W = ops.constant(W_flat, ov.Type.f32)
    matmul = ops.matmul(param_x, const_W, False, False)
    reshaped = ops.reshape(matmul, ops.constant(np.array([1, n_layers, n_experts], dtype=np.int64)), False)
    model = ov.Model([reshaped], [param_x], "moe_router_switchman")

    core = ov.Core()
    for dev in ["NPU", "GPU.0", "CPU"]:
        if dev not in core.available_devices:
            continue
        t_c = time.time()
        compiled = core.compile_model(model, dev)
        comp_s = time.time() - t_c
        req = compiled.create_infer_request()
        x_in = np.random.randn(1, d_model).astype(np.float32)
        for _ in range(20):
            req.infer({0: x_in})
        iters = 1000
        t_r = time.perf_counter()
        for _ in range(iters):
            req.infer({0: x_in})
        dt = time.perf_counter() - t_r
        us_all = (dt / iters) * 1e6
        print(f"  {dev:6s} ({core.get_property(dev, 'FULL_DEVICE_NAME')}): {us_all:6.1f} us/token for all {n_layers} layers ({us_all/n_layers:5.2f} us/layer, {iters/dt:,.1f} tok/s)")


if __name__ == "__main__":
    main()
