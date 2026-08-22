"""
export_model.py - YOLO to OpenVINO Model Exporter
===================================================
Exports the YOLO26n model to OpenVINO IR (.xml / .bin) format for
hardware-accelerated inference on Intel Iris Xe GPU / CPU.

Usage::

    python export_model.py             # export if not already done
    python export_model.py --force     # force re-export
    python export_model.py --half      # export as FP16 (better for GPU)
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import DEFAULT_CONFIG
from logger import get_logger

logger = get_logger(__name__)


def export_yolo_to_openvino(
    model_name: str = DEFAULT_CONFIG.MODEL_NAME,
    output_dir: Path = DEFAULT_CONFIG.OPENVINO_MODEL_DIR,
    imgsz: int = DEFAULT_CONFIG.INFERENCE_SIZE,
    force: bool = False,
    half: bool = False,
) -> Path:
    """
    Export a YOLO model to OpenVINO IR format.

    Args:
        model_name: Source .pt weights filename.
        output_dir: Target directory for .xml / .bin files.
        imgsz: Input image size for the exported model.
        force: Re-export even if model files already exist.
        half: Export as FP16 (recommended for Intel GPU).

    Returns:
        Path to the directory containing the exported model.
    """
    output_dir = Path(output_dir)

    # Check for existing model
    xml_files = list(output_dir.glob("*.xml")) if output_dir.exists() else []
    bin_files = list(output_dir.glob("*.bin")) if output_dir.exists() else []

    if xml_files and bin_files and not force:
        logger.info("OpenVINO model already exists in '%s' - skipping export.", output_dir)
        return output_dir

    logger.info(
        "Exporting '%s' -> OpenVINO IR (imgsz=%d, half=%s)...",
        model_name, imgsz, half,
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    # Load model (with fallback)
    model = None
    active_name = model_name
    try:
        model = YOLO(model_name)
    except Exception as exc:
        logger.warning("Could not load '%s': %s", model_name, exc)
        fallback = DEFAULT_CONFIG.FALLBACK_MODEL_NAME
        logger.info("Falling back to '%s'...", fallback)
        try:
            model = YOLO(fallback)
            active_name = fallback
        except Exception as exc2:
            logger.error("Failed to load fallback '%s': %s", fallback, exc2)
            sys.exit(1)

    # Run export
    try:
        exported = Path(
            model.export(format="openvino", imgsz=imgsz, half=half)
        )
        logger.info("Ultralytics exported to: %s", exported)

        output_dir.mkdir(parents=True, exist_ok=True)
        if exported != output_dir and exported.is_dir():
            for item in exported.glob("*"):
                shutil.copy2(item, output_dir / item.name)

        logger.info("OpenVINO model stored in: %s", output_dir)
        return output_dir

    except Exception as exc:
        logger.error("Model export failed: %s", exc)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO model to OpenVINO IR format."
    )
    parser.add_argument(
        "--model", default=DEFAULT_CONFIG.MODEL_NAME,
        help=f"Source weights file (default: {DEFAULT_CONFIG.MODEL_NAME})",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_CONFIG.OPENVINO_MODEL_DIR),
        help="Target folder for IR files",
    )
    parser.add_argument("--force", action="store_true", help="Force re-export")
    parser.add_argument(
        "--half", action="store_true", default=True,
        help="Export as FP16 (recommended for Intel GPU, default: True)",
    )
    args = parser.parse_args()

    export_yolo_to_openvino(
        model_name=args.model,
        output_dir=Path(args.output),
        imgsz=DEFAULT_CONFIG.INFERENCE_SIZE,
        force=args.force,
        half=args.half,
    )


if __name__ == "__main__":
    main()
