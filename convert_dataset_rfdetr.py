import os, argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

# 1. Demultiplex 16-band raw mosaic image into a cube
def X2Cube(img, cell_size=4):
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
    selected = cube[:, :, bands].astype(np.float32)
    for c in range(3):
        ch = selected[:, :, c]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            selected[:, :, c] = ((ch - vmin) / (vmax - vmin) * 255)
    return selected.astype(np.uint8)


def merge_to_stack_pseudo(se_rgb: np.ndarray, sa_rgb: np.ndarray) -> np.ndarray:
    """Merge SE (visible) and SA (infrared) pseudo-RGB images into a single
    3-channel stack image that stock RF-DETR (3-ch) can consume:

        ch0 = luminance of SE  (mean of R,G,B channels)
        ch1 = luminance of SA  (mean of IR R,G,B channels)
        ch2 = absolute difference |ch0 - ch1|  (highlights regions where
              visible and infrared diverge, e.g. camouflaged or heat-emitting objects)

    All channels are uint8 [0, 255].
    """
    se_gray = se_rgb.mean(axis=2).astype(np.float32)   # (H, W)
    sa_gray = sa_rgb.mean(axis=2).astype(np.float32)   # (H, W)
    diff    = np.abs(se_gray - sa_gray)                # (H, W)

    # Normalise diff to [0, 255] to maximise dynamic range
    d_min, d_max = diff.min(), diff.max()
    if d_max > d_min:
        diff = (diff - d_min) / (d_max - d_min) * 255.0

    stack = np.stack([
        se_gray.clip(0, 255).astype(np.uint8),
        sa_gray.clip(0, 255).astype(np.uint8),
        diff.clip(0, 255).astype(np.uint8),
    ], axis=2)   # (H, W, 3)
    return stack

# 2. Parse VOC XML annotation to COCO annotation records
def parse_xml_to_coco_boxes(xml_path, image_id, class_to_id, start_ann_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    img_w = float(size.find('width').text)
    img_h = float(size.find('height').text)

    annotations = []
    ann_id = start_ann_id

    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in class_to_id:
            continue
        category_id = class_to_id[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # COCO bbox: [x_min, y_min, width, height]
        bbox_w = max(0.0, xmax - xmin)
        bbox_h = max(0.0, ymax - ymin)
        area = bbox_w * bbox_h

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(xmin, 2), round(ymin, 2), round(bbox_w, 2), round(bbox_h, 2)],
            "area": round(area, 2),
            "iscrowd": 0,
            "segmentation": []
        })
        ann_id += 1

    return annotations, img_w, img_h, ann_id

def init_coco_dict(classes):
    return {
        "info": {
            "description": "Converted HSI Two-Stream Dataset in COCO format",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": i,
                "name": cls_name,
                "supercategory": "none"
            }
            for i, cls_name in enumerate(classes)
        ]
    }

def convert_test_dataset(
    raw_root="bin/raw",
    output_root="datasets/hod_converted",
    cell_size=4,
    coco_format=True
):
    raw_root = Path(raw_root)
    output_root = Path(output_root).resolve()

    test_img_dir = raw_root / "data_test/data_test/VIS"
    test_xml_dir = raw_root / "data_test/data_test/Annotations/VIS"

    classes_file = raw_root / "class.txt"
    classes = []
    class_to_id = {}
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        class_to_id = {cls: i for i, cls in enumerate(classes)}

    test_imgs = sorted(list(test_img_dir.glob("*.png")))
    if not test_imgs:
        print(f"No test images found in {test_img_dir}")
        return

    streams = ['se_information', 'sa_information', 'stack_information']
    for stream in streams:
        (output_root / stream / "test").mkdir(parents=True, exist_ok=True)

    coco_data = init_coco_dict(classes)
    ann_id = 1

    print(f"Total test samples to convert: {len(test_imgs)}")
    print("Processing test set...")

    for i, img_path in enumerate(test_imgs, start=1):
        stem = img_path.stem
        file_name = f"{stem}.png"

        # Read raw PNG and extract bands
        raw_img = np.array(Image.open(img_path))
        cube = X2Cube(raw_img, cell_size=cell_size)

        # Stream 1 (SE): bands [0, 1, 2]
        se_rgb = cube_to_pseudo_rgb(cube, bands=[0, 1, 2])
        # Stream 2 (SA): bands [5, 8, 13]
        sa_rgb = cube_to_pseudo_rgb(cube, bands=[5, 8, 13])
        # Stack: merge SE + SA into a synthetic 3-ch image
        stack_img = merge_to_stack_pseudo(se_rgb, sa_rgb)

        h, w = se_rgb.shape[:2]

        # Save images into respective stream test folders
        Image.fromarray(se_rgb).save(output_root / "se_information/test" / file_name)
        Image.fromarray(sa_rgb).save(output_root / "sa_information/test" / file_name)
        Image.fromarray(stack_img).save(output_root / "stack_information/test" / file_name)

        # Register image info
        coco_data["images"].append({
            "id": i,
            "file_name": file_name,
            "width": int(w),
            "height": int(h)
        })

        # Check if XML annotation exists for test
        xml_path = test_xml_dir / f"{stem}.xml" if test_xml_dir.exists() else None
        if xml_path and xml_path.exists():
            anns, _, _, ann_id = parse_xml_to_coco_boxes(xml_path, i, class_to_id, ann_id)
            coco_data["annotations"].extend(anns)

        if i % 100 == 0 or i == len(test_imgs):
            print(f"Processed {i}/{len(test_imgs)} test images")

    # Save COCO json files (both Roboflow and standard names for maximum compatibility)
    for stream in streams:
        stream_test_dir = output_root / stream / "test"
        with open(stream_test_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco_data, f, indent=2)
        with open(stream_test_dir / "annotations.json", "w") as f:
            json.dump(coco_data, f, indent=2)

    print(f"Test dataset conversion completed: {len(test_imgs)} samples saved.")


def convert_dataset(
    raw_root="bin/raw",
    output_root="datasets/hod_converted",
    val_ratio=0.1,
    seed=42,
    convert_test=True
):
    random.seed(seed)
    raw_root = Path(raw_root)
    output_root = Path(output_root).resolve()

    # Load classes
    classes_file = raw_root / "class.txt"
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    class_to_id = {cls: i for i, cls in enumerate(classes)}

    train_img_dir = raw_root / "data_train/data_train/VIS"
    xml_dir = raw_root / "data_train/data_train/Annotations/VIS"

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    file_stems = [f.stem for f in xml_files if (train_img_dir / f"{f.stem}.png").exists()]
    random.shuffle(file_stems)

    split_idx = int(len(file_stems) * (1 - val_ratio))
    # Roboflow and standard COCO split conventions: train, valid (or val)
    splits = {
        'train': file_stems[:split_idx],
        'valid': file_stems[split_idx:]
    }

    streams = ['se_information', 'sa_information', 'stack_information']
    for stream in streams:
        for split in ['train', 'valid']:
            (output_root / stream / split).mkdir(parents=True, exist_ok=True)

    print(f"Total labeled samples: {len(file_stems)} (train: {len(splits['train'])}, valid: {len(splits['valid'])})")

    for split, stems in splits.items():
        print(f"Processing {split} set...")
        coco_data = init_coco_dict(classes)
        ann_id = 1

        for img_id, stem in enumerate(stems, start=1):
            file_name = f"{stem}.png"

            # 1. Read raw PNG and extract bands
            img_path = train_img_dir / file_name
            raw_img = np.array(Image.open(img_path))
            cube = X2Cube(raw_img, cell_size=4)

            # Stream 1 (SE): bands [0, 1, 2]
            se_rgb = cube_to_pseudo_rgb(cube, bands=[0, 1, 2])
            # Stream 2 (SA): bands [5, 8, 13]
            sa_rgb = cube_to_pseudo_rgb(cube, bands=[5, 8, 13])
            # Stack: merge SE + SA into a synthetic 3-ch image
            stack_img = merge_to_stack_pseudo(se_rgb, sa_rgb)

            h, w = se_rgb.shape[:2]

            # Save SE, SA & stack images
            Image.fromarray(se_rgb).save(output_root / "se_information" / split / file_name)
            Image.fromarray(sa_rgb).save(output_root / "sa_information" / split / file_name)
            Image.fromarray(stack_img).save(output_root / "stack_information" / split / file_name)

            # 2. Register Image info in COCO
            coco_data["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": int(w),
                "height": int(h)
            })

            # 3. Parse XML annotations
            xml_path = xml_dir / f"{stem}.xml"
            anns, _, _, ann_id = parse_xml_to_coco_boxes(xml_path, img_id, class_to_id, ann_id)
            coco_data["annotations"].extend(anns)

        # Save COCO json files in all streams (same annotations — only images differ)
        for stream in streams:
            split_dir = output_root / stream / split
            with open(split_dir / "_annotations.coco.json", "w") as f:
                json.dump(coco_data, f, indent=2)
            with open(split_dir / "annotations.json", "w") as f:
                json.dump(coco_data, f, indent=2)

    if convert_test:
        convert_test_dataset(raw_root=raw_root, output_root=output_root, cell_size=4)

    # 3. Create dataset YAML file for all stream references
    data_yaml = {
        'train_rgb':   str(output_root / "se_information/train/"),
        'val_rgb':     str(output_root / "se_information/valid/"),
        'test_rgb':    str(output_root / "se_information/test/"),
        'train_ir':    str(output_root / "sa_information/train/"),
        'val_ir':      str(output_root / "sa_information/valid/"),
        'test_ir':     str(output_root / "sa_information/test/"),
        'train_stack': str(output_root / "stack_information/train/"),
        'val_stack':   str(output_root / "stack_information/valid/"),
        'test_stack':  str(output_root / "stack_information/test/"),
        'nc':    len(classes),
        'names': classes
    }

    yaml_path = Path("data/hsi/custom_hod_coco.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Done! Dataset converted to COCO style and YAML saved to {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert raw HSI dataset to COCO two-stream format")
    parser.add_argument("--raw-root", type=str, default="bin/raw", help="Path to raw data directory")
    parser.add_argument("--output-root", type=str, default="datasets/hod_converted", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "train", "test"], help="Conversion mode")
    args = parser.parse_args()

    if args.mode == "test":
        convert_test_dataset(raw_root=args.raw_root, output_root=args.output_root)
    elif args.mode == "train":
        convert_dataset(raw_root=args.raw_root, output_root=args.output_root, val_ratio=args.val_ratio, convert_test=False)
    else:
        convert_dataset(raw_root=args.raw_root, output_root=args.output_root, val_ratio=args.val_ratio, convert_test=True)