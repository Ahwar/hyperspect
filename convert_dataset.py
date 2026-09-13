import os
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

# 2. Convert VOC XML annotation to YOLO TXT format
def parse_xml_to_yolo(xml_path, class_to_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in class_to_id:
            continue
        cls_id = class_to_id[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Normalize to center x, center y, width, height in [0, 1]
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return yolo_lines

def convert_dataset(
    raw_root="bin/raw",
    output_root="datasets/hod_converted",
    val_ratio=0.1,
    seed=42
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
    splits = {
        'train': file_stems[:split_idx],
        'val': file_stems[split_idx:]
    }

    # Prepare directories
    streams = ['se_information', 'sa_information']
    for stream in streams:
        for split in ['train', 'val']:
            (output_root / stream / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_root / stream / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Total labeled samples: {len(file_stems)} (train: {len(splits['train'])}, val: {len(splits['val'])})")

    for split, stems in splits.items():
        print(f"Processing {split} set...")
        for stem in stems:
            # 1. Parse XML
            xml_path = xml_dir / f"{stem}.xml"
            yolo_lines = parse_xml_to_yolo(xml_path, class_to_id)
            label_text = "\n".join(yolo_lines)

            # 2. Read raw PNG and extract bands
            img_path = train_img_dir / f"{stem}.png"
            raw_img = np.array(Image.open(img_path))
            cube = X2Cube(raw_img, cell_size=4)

            # Stream 1 (SE): bands [0, 1, 2]
            se_rgb = cube_to_pseudo_rgb(cube, bands=[0, 1, 2])
            # Stream 2 (SA): bands [5, 8, 13]
            sa_rgb = cube_to_pseudo_rgb(cube, bands=[5, 8, 13])

            # Save SE
            Image.fromarray(se_rgb).save(output_root / "se_information/images" / split / f"{stem}.png")
            with open(output_root / "se_information/labels" / split / f"{stem}.txt", "w") as f:
                f.write(label_text)

            # Save SA
            Image.fromarray(sa_rgb).save(output_root / "sa_information/images" / split / f"{stem}.png")
            with open(output_root / "sa_information/labels" / split / f"{stem}.txt", "w") as f:
                f.write(label_text)

    # 3. Create dataset YAML file for train.py
    data_yaml = {
        'train_rgb': str(output_root / "se_information/images/train/"),
        'val_rgb': str(output_root / "se_information/images/val/"),
        'train_ir': str(output_root / "sa_information/images/train/"),
        'val_ir': str(output_root / "sa_information/images/val/"),
        'nc': len(classes),
        'names': classes
    }

    yaml_path = Path("data/hsi/custom_hod.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"Done! Dataset YAML saved to {yaml_path}")

if __name__ == '__main__':
    convert_dataset()