"""Inference time per photo and memory of the packaged pipeline (detection + severity, as the app calls it).

Runs the full pipeline on original photos of the validation split: one warm-up photo, then --n timed photos.
Reports seconds per photo (mean / median / min / max), the process memory after loading the models and the peak
memory during inference.
Usage: python evaluation/scripts/benchmark.py --device cpu [--n 20] [--threads N] [--out results/benchmark_v2_cpu.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))


def peak_mb(p: psutil.Process) -> float:
    m = p.memory_info()
    if hasattr(m, "peak_wset"):
        return m.peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "inference" / "weights"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from khota.pipeline import KhotaPipeline
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2**20
    t0 = time.perf_counter()
    pipe = KhotaPipeline.from_dir(a.weights, device=a.device)
    load_s = time.perf_counter() - t0
    rss_loaded = proc.memory_info().rss / 2**20

    raw = {r["image_id"]: ROOT / "data" / "raw" / f"batch{r['batch']}" / "train" / "images" / r["file"]
           for r in csv.DictReader(open(ROOT / "data" / "annotations" / "images_final.csv", encoding="utf-8"))}
    ids = sorted(p.stem for p in (ROOT / "data" / "yolo" / a.split / "images").glob("*.jpg"))[: a.n + 1]
    pipe.predict(str(raw[ids[0]]))  # warm-up
    times, n_def = [], []
    for i in ids[1:]:
        t = time.perf_counter()
        r = pipe.predict(str(raw[i]))
        if a.device != "cpu":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        n_def.append(len(r["defects"]))
    res = {
        "model_version": json.loads((Path(a.weights) / "pipeline.json").read_text()).get("model_version"),
        "device": a.device, "photos_timed": len(times), "split": a.split,
        "seconds_per_photo": {"mean": statistics.mean(times), "median": statistics.median(times),
                              "min": min(times), "max": max(times)},
        "defects_per_photo_mean": statistics.mean(n_def),
        "model_load_seconds": load_s,
        "memory_mb": {"process_before_load": rss0, "after_loading_models": rss_loaded, "peak_during_inference": peak_mb(proc)},
        "photo_size_px": "original photos (typically 4032 x 3024), as sent by the app",
    }
    out = Path(a.out or ROOT / "results" / f"benchmark_v2_{a.device}.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
