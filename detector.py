"""
detector.py - OpenVINO Accelerated YOLO Object Detector
========================================================
Runs real-time object detection using Intel OpenVINO Runtime targeting
the Intel Iris Xe integrated GPU with automatic CPU fallback.

Returns clean, structured detection dictionaries decoupled from the
underlying deep-learning framework.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from config import Config, DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing utility
# ---------------------------------------------------------------------------
def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad *im* to *new_shape* while preserving aspect ratio.

    Returns:
        (letterboxed_image, scale_ratio, (pad_w, pad_h))
    """
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, r, (dw, dh)


def classify_region(
    cx: int, frame_w: int, left_ratio: float, right_ratio: float
) -> str:
    """Return ``'left'``, ``'center'``, or ``'right'`` for a centre point."""
    if cx < frame_w * left_ratio:
        return "left"
    if cx > frame_w * right_ratio:
        return "right"
    return "center"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class OpenVINODetector:
    """
    Object detector powered by Intel OpenVINO Runtime and YOLO.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        device: Optional[str] = None,
        auto_export_if_missing: bool = True,
    ):
        self.config = config
        self.requested_device = (device or config.OPENVINO_DEVICE).upper()
        self.active_device: str = "CPU"
        self.device_full_name: str = "Intel CPU"
        self.core = None
        self.compiled_model = None
        self.output_layer = None
        self.classes: Dict[int, str] = {}
        self.model_dir = Path(config.OPENVINO_MODEL_DIR)
        self.xml_path: Optional[Path] = None

        # Timing (exposed for HUD)
        self.last_inference_ms: float = 0.0

        self._ensure_model_exists(auto_export_if_missing)
        self._load_class_names()
        self._init_openvino_runtime()

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------
    def _ensure_model_exists(self, auto_export: bool) -> None:
        """Verify OpenVINO IR files exist, optionally triggering export."""
        xml_files = (
            list(self.model_dir.glob("*.xml")) if self.model_dir.exists() else []
        )
        if not xml_files:
            if auto_export:
                logger.info(
                    "OpenVINO model not found at '%s' - exporting now...",
                    self.model_dir,
                )
                from export_model import export_yolo_to_openvino

                self.model_dir = export_yolo_to_openvino(
                    model_name=self.config.MODEL_NAME,
                    output_dir=self.config.OPENVINO_MODEL_DIR,
                    imgsz=self.config.INFERENCE_SIZE,
                )
                xml_files = list(self.model_dir.glob("*.xml"))
            else:
                raise FileNotFoundError(
                    f"OpenVINO model not found at '{self.model_dir}'. "
                    "Run:  python export_model.py"
                )
        if not xml_files:
            raise FileNotFoundError(
                f"No .xml model file in '{self.model_dir}'."
            )
        self.xml_path = xml_files[0]

    def _load_class_names(self) -> None:
        """Load class-name map from ``metadata.yaml`` or fall back to COCO80."""
        meta = self.model_dir / "metadata.yaml"
        if meta.exists():
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", {})
                    self.classes = {int(k): str(v) for k, v in names.items()}
                    logger.info(
                        "Loaded %d class names from metadata.yaml",
                        len(self.classes),
                    )
            except Exception as exc:
                logger.warning("Could not parse metadata.yaml: %s", exc)

        if not self.classes:
            self.classes = _coco80_classes()
            logger.info("Using built-in COCO-80 class names")

    def _init_openvino_runtime(self) -> None:
        """Initialise OpenVINO Core, compile model on GPU (fallback CPU)."""
        import openvino as ov

        self.core = ov.Core()
        available = self.core.available_devices

        logger.info("OpenVINO devices: %s", ", ".join(available))

        gpu_devs = [d for d in available if d.startswith("GPU")]
        has_gpu = bool(gpu_devs)

        ov_model = self.core.read_model(str(self.xml_path))

        # Choose target device
        target = "CPU"
        if self.requested_device in ("GPU", "AUTO") and has_gpu:
            target = gpu_devs[0]
        elif self.requested_device == "CPU":
            target = "CPU"

        logger.info(
            "Compiling model '%s' on device '%s'...",
            self.xml_path.name,
            target,
        )
        try:
            self.compiled_model = self.core.compile_model(ov_model, target)
            self.active_device = target
            try:
                self.device_full_name = self.core.get_property(
                    target, "FULL_DEVICE_NAME"
                )
            except Exception:
                self.device_full_name = target
            logger.info(
                "Model compiled on %s (%s)",
                self.active_device,
                self.device_full_name,
            )
        except Exception as exc:
            if target != "CPU":
                logger.warning(
                    "GPU compilation failed (%s) - falling back to CPU", exc
                )
                self.compiled_model = self.core.compile_model(ov_model, "CPU")
                self.active_device = "CPU"
                try:
                    self.device_full_name = self.core.get_property(
                        "CPU", "FULL_DEVICE_NAME"
                    )
                except Exception:
                    self.device_full_name = "Intel CPU"
                logger.info("CPU fallback OK (%s)", self.device_full_name)
            else:
                raise RuntimeError(
                    f"Failed to compile model on CPU: {exc}"
                ) from exc

        self.output_layer = self.compiled_model.outputs[0]

        # Warm-up inference
        dummy = np.zeros(
            (1, 3, self.config.INFERENCE_SIZE, self.config.INFERENCE_SIZE),
            dtype=np.float32,
        )
        _ = self.compiled_model([dummy])[self.output_layer]
        logger.info("Warm-up inference complete")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Letterbox + BGR->RGB + HWC->CHW + normalise to [0, 1]."""
        sz = self.config.INFERENCE_SIZE
        img, ratio, (pw, ph) = letterbox(frame, new_shape=(sz, sz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # -> BCHW
        return img, ratio, (pw, ph)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on *frame* and return structured detections::

            [
                {
                    "class_name":  "person",
                    "class_id":    0,
                    "confidence":  0.91,
                    "bbox":        [x1, y1, x2, y2],
                    "center":      [cx, cy],
                    "width":       w,
                    "height":      h,
                    "area_ratio":  0.12,
                    "region":      "center",
                },
                ...
            ]
        """
        if frame is None or self.compiled_model is None:
            return []

        fh, fw = frame.shape[:2]
        total_px = float(fw * fh)

        # 1. Preprocess
        tensor, ratio, (pad_w, pad_h) = self.preprocess(frame)

        # 2. Inference
        import time as _t

        t0 = _t.perf_counter()
        out = self.compiled_model([tensor])[self.output_layer]
        self.last_inference_ms = (_t.perf_counter() - t0) * 1000.0

        # 3. Parse output - supports both End-to-End [N, 6] and anchor grid [84, 8400]
        if out is None or out.ndim < 2:
            return []

        batch = out[0] if out.ndim == 3 else out
        candidate_boxes = []
        candidate_scores = []
        candidate_cids = []

        conf_thresh = getattr(self.config, "CONFIDENCE_THRESHOLD", 0.30)

        # Format A: End-to-End [N, 6] (x1, y1, x2, y2, score, cid)
        if batch.ndim == 2 and batch.shape[1] == 6:
            for row in batch:
                x1r, y1r, x2r, y2r, score, cid_raw = row[:6]
                conf = float(score)
                if conf < conf_thresh:
                    continue
                cid = int(cid_raw)
                x1 = int(max(0, min(fw - 1, round((x1r - pad_w) / ratio))))
                y1 = int(max(0, min(fh - 1, round((y1r - pad_h) / ratio))))
                x2 = int(max(0, min(fw - 1, round((x2r - pad_w) / ratio))))
                y2 = int(max(0, min(fh - 1, round((y2r - pad_h) / ratio))))
                if x2 <= x1 or y2 <= y1:
                    continue
                candidate_boxes.append([x1, y1, x2 - x1, y2 - y1])
                candidate_scores.append(conf)
                candidate_cids.append(cid)

        # Format B: Standard YOLO anchor grid output [84, 8400] or [8400, 84] (cx, cy, w, h, class_probs...)
        elif batch.ndim == 2 and (batch.shape[0] >= 5 or batch.shape[1] >= 5):
            pred = batch if (batch.shape[1] >= 5 and batch.shape[0] > batch.shape[1]) else batch.T
            # pred is now (8400, 84)
            boxes = pred[:, :4]
            scores_matrix = pred[:, 4:]

            max_scores = np.max(scores_matrix, axis=1)
            class_ids = np.argmax(scores_matrix, axis=1)

            mask = max_scores >= conf_thresh
            if np.any(mask):
                valid_boxes = boxes[mask]
                valid_scores = max_scores[mask]
                valid_cids = class_ids[mask]

                for i in range(len(valid_boxes)):
                    cx_r, cy_r, w_r, h_r = valid_boxes[i]
                    x1r = cx_r - w_r / 2.0
                    y1r = cy_r - h_r / 2.0
                    x2r = cx_r + w_r / 2.0
                    y2r = cy_r + h_r / 2.0

                    x1 = int(max(0, min(fw - 1, round((x1r - pad_w) / ratio))))
                    y1 = int(max(0, min(fh - 1, round((y1r - pad_h) / ratio))))
                    x2 = int(max(0, min(fw - 1, round((x2r - pad_w) / ratio))))
                    y2 = int(max(0, min(fh - 1, round((y2r - pad_h) / ratio))))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    candidate_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    candidate_scores.append(float(valid_scores[i]))
                    candidate_cids.append(int(valid_cids[i]))

        if not candidate_boxes:
            return []

        # Apply class-aware Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes of the SAME class.
        # By offsetting boxes by class_id * 4096, boxes of DIFFERENT classes (e.g. person holding a cell phone,
        # or person sitting on a chair) will never overlap in coordinate space and are preserved!
        nms_thresh = getattr(self.config, "NMS_THRESHOLD", 0.45)
        offset_boxes = [
            [bx + cid * 4096, by + cid * 4096, bw, bh]
            for (bx, by, bw, bh), cid in zip(candidate_boxes, candidate_cids)
        ]
        indices = cv2.dnn.NMSBoxes(
            offset_boxes,
            candidate_scores,
            score_threshold=conf_thresh,
            nms_threshold=nms_thresh,
        )

        results: List[Dict] = []
        if len(indices) > 0:
            indices_flat = indices.flatten() if hasattr(indices, "flatten") else [int(i) for i in indices]
            for idx in indices_flat:
                bx, by, bw, bh = candidate_boxes[idx]
                conf = candidate_scores[idx]
                cid = candidate_cids[idx]
                name = self.classes.get(cid, f"object_{cid}")

                x1 = bx
                y1 = by
                x2 = bx + bw
                y2 = by + bh
                cx = x1 + bw // 2
                cy = y1 + bh // 2
                area = bw * bh
                area_ratio = area / total_px if total_px > 0 else 0.0

                region = classify_region(
                    cx, fw, self.config.PATH_LEFT_RATIO, self.config.PATH_RIGHT_RATIO
                )

                results.append(
                    {
                        "class_name": name,
                        "class_id": cid,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "width": bw,
                        "height": bh,
                        "area_ratio": round(area_ratio, 4),
                        "region": region,
                    }
                )

        return results


# ---------------------------------------------------------------------------
# Fallback COCO-80 class names
# ---------------------------------------------------------------------------
def _coco80_classes() -> Dict[int, str]:
    names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    return {i: n for i, n in enumerate(names)}
