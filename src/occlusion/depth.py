"""Single-view DA3 adapter. Predictions have arbitrary scale, never meters."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

DEFAULT_MODEL = "depth-anything/DA3-SMALL"


@dataclass
class DepthResult:
    rgb: np.ndarray  # DA3 processed RGB, aligned pixel-for-pixel with depth
    depth: np.ndarray
    confidence: np.ndarray | None
    intrinsics: np.ndarray | None
    extrinsics: np.ndarray | None
    seconds: float


class DA3Estimator:
    def __init__(self, device="cpu", resolution=252):
        import torch
        from depth_anything_3.api import DepthAnything3

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")

        class CPUCompatibleDA3(DepthAnything3):
            # Upstream forward enables FP16 autocast on CPU. Use FP32 there;
            # preserve upstream mixed precision on CUDA (including older GPUs).
            def forward(self, image, *args, **kwargs):
                if image.device.type == "cpu":
                    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                        return self.model(image, *args, **kwargs)
                return super().forward(image, *args, **kwargs)

        self.torch = torch
        self.device = device
        self.resolution = resolution
        self.model = CPUCompatibleDA3.from_pretrained(DEFAULT_MODEL).to(device).eval()

    def predict(self, rgb):
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with self.torch.inference_mode():
            prediction = self.model.inference(
                [rgb], process_res=self.resolution,
                process_res_method="upper_bound_resize",
            )
        seconds = perf_counter() - start
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if depth.ndim != 2 or not np.any(np.isfinite(depth) & (depth > 0)):
            raise RuntimeError("Model returned no valid positive depth")
        processed = np.asarray(prediction.processed_images[0], dtype=np.uint8)
        if processed.shape != (*depth.shape, 3):
            raise RuntimeError("Processed RGB and predicted depth are not aligned")

        def first(name):
            value = getattr(prediction, name, None)
            return None if value is None else np.asarray(value[0])

        return DepthResult(processed, depth, first("conf"), first("intrinsics"),
                           first("extrinsics"), seconds)

    def peak_cuda_bytes(self):
        return self.torch.cuda.max_memory_allocated() if self.device == "cuda" else None
