"""
tools/benchmark.py - Inference Benchmark
==========================================
Measures CPU vs GPU inference latency and throughput.

Run::

    python tools/benchmark.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from config import Config
from logger import get_logger

logger = get_logger("benchmark")

WARMUP_FRAMES = 10
BENCHMARK_FRAMES = 50


def benchmark_device(device: str, config: Config) -> dict:
    """Benchmark inference on *device*.  Returns stats dict."""
    import openvino as ov
    from pathlib import Path

    model_dir = Path(config.OPENVINO_MODEL_DIR)
    xml_files = list(model_dir.glob("*.xml"))
    if not xml_files:
        return {"error": f"No model found in {model_dir}"}

    core = ov.Core()
    available = core.available_devices

    if device.startswith("GPU") and not any(d.startswith("GPU") for d in available):
        return {"error": "GPU not available"}

    try:
        model = core.read_model(str(xml_files[0]))
        compiled = core.compile_model(model, device)
        output = compiled.outputs[0]
    except Exception as exc:
        return {"error": str(exc)}

    sz = config.INFERENCE_SIZE
    dummy = np.random.rand(1, 3, sz, sz).astype(np.float32)

    # Warm up
    for _ in range(WARMUP_FRAMES):
        _ = compiled([dummy])[output]

    # Benchmark
    times = []
    for _ in range(BENCHMARK_FRAMES):
        t0 = time.perf_counter()
        _ = compiled([dummy])[output]
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "device": device,
        "avg_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
        "fps": float(1000.0 / np.mean(arr)),
    }


def run_full_benchmark(frames: int = BENCHMARK_FRAMES) -> None:
    global BENCHMARK_FRAMES
    BENCHMARK_FRAMES = frames

    config = Config()

    print("\n" + "=" * 55)
    print("        BLIND CAP - INFERENCE BENCHMARK")
    print("=" * 55)

    for device in ("CPU", "GPU"):
        print(f"\n  Benchmarking {device}...")
        result = benchmark_device(device, config)
        if "error" in result:
            print(f"    [FAIL] {result['error']}")
        else:
            print(f"    Avg: {result['avg_ms']:.1f} ms")
            print(f"    Min: {result['min_ms']:.1f} ms")
            print(f"    Max: {result['max_ms']:.1f} ms")
            print(f"    Std: {result['std_ms']:.1f} ms")
            print(f"    FPS: {result['fps']:.1f}")

    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    run_full_benchmark()
