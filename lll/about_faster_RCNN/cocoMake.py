import os, json, cv2
from glob import glob

root_dir = "A_tvt"

classes = [
    'crackdown', 
    'crossline', 
    'danger', 
    'kidprotectzone', 
    'leftcurve', 
    'noparking', 
    'nouturn', 
    'ntrack', 
    'oneway', 
    'slipper', 
    'slow', 
    'speedbump', 
    'speedlimit', 
    'stop', 
    'turncross', 
    'uturn'
    ]


def yolo_to_coco(img_dir, label_dir, output_json, classes):
    images, annotations, categories = [], [], []
    ann_id = 1

    for i, cls in enumerate(classes):
        categories.append({"id": i, "name": cls, "supercategory": "none"})

    img_files = sorted(
        glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        +glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
    )

    print(f" Found {len(img_files)} images in {img_dir}")

    for img_id, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            bw *= w
            bh *= h

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)
    print(f" {output_json} saved ({len(images)} images, {len(annotations)} annotations)")

splits = ["train", "val", "test"]
for split in splits:
    img_dir = os.path.join(root_dir, split, "images")
    label_dir = os.path.join(root_dir, split, "labels")
    output_json = os.path.join(root_dir, f"{split}.json")
    yolo_to_coco(img_dir, label_dir, output_json, classes)
