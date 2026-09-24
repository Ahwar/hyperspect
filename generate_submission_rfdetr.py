"""Generate submission.csv for RF-DETR model evaluation and leaderboard submission.

Supports flexible stream-modes matching train_rfdetr.py:
- 'rgb': RGB / SE information stream (3 channels)
- 'ir': IR / SA information stream (3 channels)
- 'stack': Stacked/pseudo stream (3 channels)
- 'dual': Two-stream stacked channel input (6 channels)

Accepts either:
1. Converted dataset test directories (e.g., datasets/hod_converted/se_information/test or .yaml)
2. Raw 16-band mosaic test images directly (.png) with automated demultiplexing / pseudo-RGB conversion
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# 16-band HSI Demultiplexing and Preprocessing Helpers
# ------------------------------------------------------------------------
def X2Cube(img: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Demultiplex 16-band raw mosaic image into a cube of shape (H/4, W/4, 16)."""
    B = [cell_size, cell_size]
    skip = [cell_size, cell_size]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    return out.reshape(M // cell_size, N // cell_size, cell_size * cell_size)


def cube_to_pseudo_rgb(cube: np.ndarray, bands: list[int]) -> np.ndarray:
    """Convert selected 3 bands of cube into 3-channel uint8 pseudo-RGB image."""
    selected = cube[:, :, bands].astype(np.float32)
    for c in range(3):
        ch = selected[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            selected[:, :, c] = ((ch - vmin) / (vmax - vmin) * 255.0)
    return selected.astype(np.uint8)


def load_and_preprocess_sample(
    img_path: Path,
    stream_mode: str = 'rgb',
    ir_path: Path | None = None
) -> Image.Image | np.ndarray:
    """Load and preprocess image sample based on stream-mode and input file type.

    Returns PIL.Image (for standard 3-channel) or np.ndarray (for 6-channel dual).
    """
    raw_cube = None

    # Check if input is a 16-band raw mosaic image (grayscale 2D uint8/uint16 with 1 channel)
    try:
        raw_arr = np.array(Image.open(img_path))
        if raw_arr.ndim == 2:
            raw_cube = X2Cube(raw_arr, cell_size=4)
    except Exception:
        raw_cube = None

    if stream_mode == 'rgb':
        if raw_cube is not None:
            arr = cube_to_pseudo_rgb(raw_cube, bands=[0, 1, 2])
            return Image.fromarray(arr)
        return Image.open(img_path).convert("RGB")

    elif stream_mode == 'ir':
        if raw_cube is not None:
            arr = cube_to_pseudo_rgb(raw_cube, bands=[5, 8, 13])
            return Image.fromarray(arr)
        if ir_path is not None and ir_path.exists():
            return Image.open(ir_path).convert("RGB")
        return Image.open(img_path).convert("RGB")

    elif stream_mode == 'stack':
        if raw_cube is not None:
            arr = cube_to_pseudo_rgb(raw_cube, bands=[0, 1, 2])
            return Image.fromarray(arr)
        return Image.open(img_path).convert("RGB")

    elif stream_mode == 'dual':
        # 6-channel composite: [SE (3ch), SA (3ch)]
        if img_path.suffix.lower() == '.npy':
            return np.load(img_path)
        if raw_cube is not None:
            se_rgb = cube_to_pseudo_rgb(raw_cube, bands=[0, 1, 2])
            sa_rgb = cube_to_pseudo_rgb(raw_cube, bands=[5, 8, 13])
        else:
            se_rgb = np.array(Image.open(img_path).convert("RGB"))
            if ir_path is not None and ir_path.exists():
                sa_rgb = np.array(Image.open(ir_path).convert("RGB"))
            else:
                sa_rgb = se_rgb
        return np.concatenate([se_rgb, sa_rgb], axis=-1)

    else:
        return Image.open(img_path).convert("RGB")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for submission generation."""
    default_data = './data/hsi/custom_hod_coco.yaml' if Path('./data/hsi/custom_hod_coco.yaml').exists() else './data/hsi/custom_hod.yaml'

    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on test images and generate submission.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        dest="checkpoint",
        type=str,
        default="project/rfdetr/train/exp/checkpoint_best_ema.pth",
        help="Path to trained model checkpoint (.pth or .ckpt).",
    )
    parser.add_argument(
        "--stream-mode",
        type=str,
        default="rgb",
        choices=["rgb", "ir", "stack", "dual"],
        help="Stream strategy: rgb (3ch), ir (3ch), stack (pseudo 3ch), dual (6ch).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=default_data,
        help="Path to data YAML file (used to auto-resolve test directories and classes).",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to directory containing test images (overrides data.yaml if specified).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=576,
        help="Inference image resolution (must be divisible by 32).",
    )
    parser.add_argument(
        "--conf-threshold",
        "--conf-thres",
        dest="conf_threshold",
        type=float,
        default=0.001,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--iou-threshold",
        "--iou-thres",
        dest="iou_threshold",
        type=float,
        default=None,
        help="Optional IoU threshold for post-processing NMS (e.g. 0.5). Disabled if not set.",
    )
    parser.add_argument(
        "--class-agnostic-nms",
        action="store_true",
        help="Apply class-agnostic NMS instead of per-class NMS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda', '0', 'cpu').",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Optimize model for float16 inference using model.inference(dtype=torch.float16).",
    )
    parser.add_argument(
        "--class-id-offset",
        type=int,
        default=0,
        help="Offset added to model predicted 0-indexed class_id (default: 0 for 0-indexed classes).",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=2,
        help="Number of decimal places to round coordinates in CSV.",
    )
    return parser.parse_args()


def get_image_id(file_name: str) -> int:
    """Extract integer image ID from filename (e.g. '1009.jpg' -> 1009)."""
    stem = Path(file_name).stem
    if stem.isdigit():
        return int(stem)
    extracted = "".join(c for c in stem if c.isdigit())
    return int(extracted) if extracted else 0


def resolve_test_directories(args: argparse.Namespace) -> tuple[Path, Path | None, int, list[str]]:
    """Resolve primary and secondary test directories along with class info."""
    num_classes = 18
    class_names = [f"class_{i}" for i in range(num_classes)]

    if Path(args.data).exists():
        try:
            with open(args.data, "r") as f:
                data_dict = yaml.safe_load(f) or {}
            num_classes = int(data_dict.get("nc", num_classes))
            class_names = data_dict.get("names", class_names)

            test_rgb = data_dict.get("test_rgb") or data_dict.get("val_rgb") or ""
            test_ir = data_dict.get("test_ir") or data_dict.get("val_ir") or ""
        except Exception as e:
            logger.warning("Failed to parse data config %s: %e", args.data, e)
            test_rgb, test_ir = "", ""
    else:
        test_rgb, test_ir = "", ""

    # Check CLI explicit override
    if args.test_dir:
        test_rgb_path = Path(args.test_dir)
        test_ir_path = None
        test_dual_path = None
    else:
        test_rgb_path = Path(test_rgb) if test_rgb else Path("datasets/hod_converted/se_information/test")
        test_ir_path = Path(test_ir) if test_ir else Path("datasets/hod_converted/sa_information/test")
        test_dual = data_dict.get("test_dual") or data_dict.get("val_dual") or ""
        if test_dual:
            test_dual_path = Path(test_dual)
        else:
            test_dual_path = Path(str(test_rgb_path).replace('se_information', 'dual_information').replace('sa_information', 'dual_information'))

    # Resolve stream directory path matching train_rfdetr.py logic
    if args.stream_mode == 'rgb':
        stream_path = test_rgb_path
    elif args.stream_mode == 'ir':
        stream_path = test_ir_path if (test_ir_path and test_ir_path.exists()) else test_rgb_path
    elif args.stream_mode == 'dual':
        stream_path = test_dual_path if (test_dual_path and test_dual_path.exists()) else test_rgb_path
    elif args.stream_mode == 'stack':
        stream_path = test_rgb_path
    else:
        stream_path = test_rgb_path

    # Fallback to raw test if stream path folder doesn't exist
    if not stream_path or not stream_path.exists():
        raw_test = Path("bin/raw/data_test/data_test/VIS")
        if raw_test.exists():
            logger.info("Stream directory %s not found. Falling back to raw test: %s", stream_path, raw_test)
            stream_path = raw_test

    primary_test_path = stream_path
    secondary_test_path = test_ir_path if (args.stream_mode == 'dual' and test_ir_path and test_ir_path.exists()) else None

    return primary_test_path, secondary_test_path, num_classes, class_names



def apply_multichannel_patch(num_channels: int = 6) -> None:
    """Patch DINOv2 PatchEmbeddings and weights loading so 6-channel models can be instantiated and loaded."""
    if num_channels == 3:
        return
    import copy as _copy
    import torch
    from rfdetr.models.backbone.dinov2_with_windowed_attn import Dinov2WithRegistersPatchEmbeddings as _PatchEmbCls
    from rfdetr.models import weights as _wmod

    _target_channels = num_channels

    # Patch __init__ so every new instance gets num_channels in_channels
    _orig_patch_init = _PatchEmbCls.__init__

    def _multichannel_patch_init(self, config):
        config_copy = _copy.copy(config)
        config_copy.num_channels = _target_channels
        _orig_patch_init(self, config_copy)

    _PatchEmbCls.__init__ = _multichannel_patch_init

    # Patch load_pretrain_weights to handle channel expansion if loading a 3-ch checkpoint into 6-ch model
    _orig_lpw = _wmod.load_pretrain_weights

    def _multichannel_lpw(nn_model, model_config, trust=True):
        _orig_lsd = torch.nn.Module.load_state_dict

        def _expanded_lsd(self, state_dict, strict=True, **kwargs):
            expanded = {}
            for k, v in state_dict.items():
                if ('patch_embeddings.projection.weight' in k
                        and v.ndim == 4
                        and v.shape[1] < _target_channels):
                    reps = _target_channels // v.shape[1]
                    expanded[k] = v.repeat(1, reps, 1, 1) / reps
                else:
                    expanded[k] = v
            return _orig_lsd(self, expanded, strict=strict, **kwargs)

        torch.nn.Module.load_state_dict = _expanded_lsd
        try:
            return _orig_lpw(nn_model, model_config, trust=trust)
        finally:
            torch.nn.Module.load_state_dict = _orig_lsd

    _wmod.load_pretrain_weights = _multichannel_lpw


def load_rfdetr_model(args: argparse.Namespace) -> Any:
    """Initialize and load RF-DETR model directly from checkpoint."""
    num_channels = 6 if args.stream_mode in ['stack', 'dual'] else 3
    if num_channels != 3:
        logger.info(f"Applying patch for {num_channels}-channel input (stream_mode={args.stream_mode})...")
        apply_multichannel_patch(num_channels)

    checkpoint_path = Path(args.checkpoint)
    from rfdetr import RFDETR
    logger.info("Loading RF-DETR model from checkpoint: %s...", checkpoint_path)
    model = RFDETR.from_checkpoint(
        str(checkpoint_path),
        trust_checkpoint=True,
        device=args.device
    )
    return model



def generate_submission(args: argparse.Namespace) -> None:
    """Run model inference on test images and write formatted submission.csv."""
    primary_test_path, secondary_test_path, num_classes, _ = resolve_test_directories(args)

    if not primary_test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {primary_test_path}")

    # Gather test images sorted numerically or alphabetically
    img_files = sorted(
        [p for p in primary_test_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".npy")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )

    if not img_files:
        raise ValueError(f"No images found in test directory: {primary_test_path}")

    logger.info("Found %d test images in %s (stream-mode: %s)", len(img_files), primary_test_path, args.stream_mode)
    model = load_rfdetr_model(args)

    if args.fp16:
        import torch
        logger.info("Optimizing model for FP16 inference...")
        try:
            model.inference(dtype=torch.float16)
        except Exception as e:
            logger.warning("Could not set FP16 inference mode: %s", e)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
    row_id = 0
    total_detections = 0

    logger.info(
        "Running inference (conf_threshold=%.4f, iou_threshold=%s, batch_size=%d)...",
        args.conf_threshold,
        args.iou_threshold,
        args.batch_size,
    )

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        batch_size = max(1, args.batch_size)
        for i in tqdm(range(0, len(img_files), batch_size), desc="Inferring test images", unit="batch"):
            batch_paths = img_files[i : i + batch_size]
            batch_inputs = []

            for p in batch_paths:
                ir_p = (secondary_test_path / p.name) if secondary_test_path else None
                sample = load_and_preprocess_sample(
                    p,
                    stream_mode=args.stream_mode,
                    ir_path=ir_p
                )
                batch_inputs.append(sample)

            # Predict on batch or single sample
            preds = model.predict(
                batch_inputs if len(batch_inputs) > 1 else batch_inputs[0],
                threshold=args.conf_threshold,
                include_source_image=False,
            )

            # Standardize return to list of Detections
            if not isinstance(preds, list):
                preds = [preds]

            for img_p, dets in zip(batch_paths, preds):
                img_id = get_image_id(img_p.name)

                # Apply optional NMS
                if args.iou_threshold is not None and len(dets) > 0:
                    dets = dets.with_nms(
                        threshold=args.iou_threshold,
                        class_agnostic=args.class_agnostic_nms,
                    )

                if len(dets) == 0:
                    continue

                boxes = dets.xyxy  # (N, 4): x1, y1, x2, y2
                confidences = dets.confidence  # (N,)
                class_ids = dets.class_id + args.class_id_offset  # (N,)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    conf_formatted = f"{float(conf):.4f}"
                    x1_val = round(float(x1), args.round_decimals)
                    y1_val = round(float(y1), args.round_decimals)
                    x2_val = round(float(x2), args.round_decimals)
                    y2_val = round(float(y2), args.round_decimals)

                    writer.writerow([row_id, img_id, int(cls_id), conf_formatted, x1_val, y1_val, x2_val, y2_val])
                    row_id += 1
                    total_detections += 1

    logger.info("Submission generated successfully: %s", output_path.resolve())
    logger.info("Total rows written: %d across %d images", total_detections, len(img_files))


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    generate_submission(args)


if __name__ == "__main__":
    main()

