"""
Two-Stream YOLO Inference & Submission Generator for 16-Band HSI Dataset

This script runs inference on 16-channel hyperspectral raw images (.png) using a trained
two-stream YOLO model (S2ADet) and generates a submission CSV file matching the format:
    id,image_id,class_id,confidence,x1,y1,x2,y2

Pipeline:
1. Load 16-band raw mosaic test images from bin/raw/data_test/data_test/VIS.
2. Demultiplex into 16 spectral channels via X2Cube.
3. Extract Stream 1 (SE: bands [0, 1, 2]) and Stream 2 (SA: bands [5, 8, 13]).
4. Preprocess & letterbox images to model input size.
5. Perform two-stream inference: out, _ = model(img_rgb, img_ir).
6. Apply NMS, rescale coordinates to original image coordinates, and export CSV.

Usage Example:
    ./venv/bin/python generate_submission.py \\
        --weights runs/train/exp/weights/best.pt \\
        --test-dir bin/raw/data_test/data_test/VIS \\
        --output-csv submission.csv \\
        --conf-thres 0.25 \\
        --iou-thres 0.45 \\
        --device 0
"""

import argparse
import csv
import os
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox


def X2Cube(img, cell_size=4):
    """
    Demultiplex 16-band raw mosaic image into a cube of shape (H/4, W/4, 16)
    """
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


def cube_to_pseudo_rgb(cube, bands):
    """
    Convert selected 3 bands of cube into 3-channel uint8 pseudo-RGB image
    """
    selected = cube[:, :, bands].astype(np.float32)
    for c in range(3):
        ch = selected[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            selected[:, :, c] = ((ch - vmin) / (vmax - vmin) * 255.0)
    return selected.astype(np.uint8)


def preprocess_image(raw_img_path, imgsz=640, stride=32, auto=False):
    """
    Load raw 16-band PNG, extract SE (bands [0,1,2]) and SA (bands [5,8,13]),
    apply letterbox padding, and return RGB & IR tensors along with shape info.
    """
    raw_img = np.array(Image.open(raw_img_path))
    cube = X2Cube(raw_img, cell_size=4)

    # Stream 1 (SE): bands [0, 1, 2]
    img_rgb = cube_to_pseudo_rgb(cube, bands=[0, 1, 2])
    # Stream 2 (SA): bands [5, 8, 13]
    img_ir = cube_to_pseudo_rgb(cube, bands=[5, 8, 13])

    orig_shape = img_rgb.shape[:2]  # (H, W) of cube / pseudo-RGB image

    # Letterbox resize & padding (auto=False pads to exact imgsz x imgsz so batch stacking succeeds)
    img_rgb_padded, ratio, (dw, dh) = letterbox(img_rgb, new_shape=imgsz, auto=auto, stride=stride)
    img_ir_padded, _, _ = letterbox(img_ir, new_shape=imgsz, auto=auto, stride=stride)

    # Convert to Tensor (B, C, H, W) normalized to [0.0, 1.0]
    # Note: letterbox preserves RGB channel ordering if input is RGB
    t_rgb = torch.from_numpy(img_rgb_padded).permute(2, 0, 1).float() / 255.0
    t_ir = torch.from_numpy(img_ir_padded).permute(2, 0, 1).float() / 255.0

    return t_rgb, t_ir, orig_shape, (ratio, (dw, dh))


@torch.no_grad()
def generate_submission(
    weights,
    test_dir="bin/raw/data_test/data_test/VIS",
    output_csv="submission.csv",
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.5,
    device="0",
    batch_size=16,
    augment=False,
    half_precision=True,
    max_det=300
):
    set_logging()
    device = select_device(device, batch_size=batch_size)

    # 1. Load trained model
    print(f"Loading weights from {weights} ...")
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    imgsz = check_img_size(imgsz, s=stride)

    half = device.type != 'cpu' and half_precision
    if half:
        model.half()
    model.eval()

    # 2. Get list of test images
    test_dir = Path(test_dir)
    image_paths = sorted(list(test_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {test_dir}")

    print(f"Found {len(image_paths)} test images in {test_dir}")

    # 3. Iterate through test images in batches
    submission_rows = []
    submission_id = 0

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_rgb = []
        batch_ir = []
        batch_shapes = []

        for p in batch_paths:
            t_rgb, t_ir, orig_shape, ratio_pad = preprocess_image(p, imgsz=imgsz, stride=stride, auto=False)
            batch_rgb.append(t_rgb)
            batch_ir.append(t_ir)
            batch_shapes.append((orig_shape, ratio_pad))

        # Stack into batch tensors
        batch_rgb = torch.stack(batch_rgb, dim=0).to(device)
        batch_ir = torch.stack(batch_ir, dim=0).to(device)

        if half:
            batch_rgb = batch_rgb.half()
            batch_ir = batch_ir.half()

        # Model inference
        preds, _ = model(batch_rgb, batch_ir, augment=augment)

        # Non-Max Suppression
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)

        # Process detections per image in batch
        for idx, pred in enumerate(preds):
            img_path = batch_paths[idx]
            image_id = img_path.stem  # e.g. '1000'
            orig_shape, ratio_pad = batch_shapes[idx]

            if len(pred) > 0:
                if len(pred) > max_det:
                    pred = pred[:max_det]

                # Rescale boxes back to original pseudo-RGB image coordinates
                pred_coords = pred[:, :4].clone()
                scale_coords(batch_rgb[idx].shape[1:], pred_coords, orig_shape, ratio_pad=ratio_pad)

                for k in range(len(pred)):
                    x1, y1, x2, y2 = pred_coords[k].tolist()
                    conf = float(pred[k, 4].item())
                    cls_id = int(pred[k, 5].item())

                    submission_rows.append({
                        'id': submission_id,
                        'image_id': image_id,
                        'class_id': cls_id,
                        'confidence': round(conf, 4),
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    })
                    submission_id += 1

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(image_paths):
            print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...")

    # 4. Save to CSV
    fieldnames = ['id', 'image_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2']
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(submission_rows)
    print(f"Successfully generated submission file with {len(submission_rows)} detections at: {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate submission CSV from 16-channel test images using two-stream YOLO model")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights checkpoint (.pt)')
    parser.add_argument('--test-dir', type=str, default='bin/raw/data_test/data_test/VIS', help='Path to test images folder')
    parser.add_argument('--output-csv', type=str, default='submission.csv', help='Path to save output submission.csv')
    parser.add_argument('--img-size', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold (e.g., 0.25 or 0.001)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections per image')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='Augmented inference')
    parser.add_argument('--no-half', action='store_true', help='Do not use half precision (FP16)')

    args = parser.parse_args()

    generate_submission(
        weights=args.weights,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        imgsz=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        device=args.device,
        batch_size=args.batch_size,
        augment=args.augment,
        half_precision=not args.no_half
    )
