import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_files(image_dir):
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def matching_mask_path(image_path, mask_dir):
    mask_dir = Path(mask_dir)
    candidates = [
        mask_dir / f"{image_path.stem}_pseudo.png",
        mask_dir / f"{image_path.stem}_pseudo.jpg",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}.jpg",
        mask_dir / image_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def padded_square_box(x, y, w, h, image_width, image_height, padding):
    cx = x + (w / 2)
    cy = y + (h / 2)
    side = int(max(w, h) + (padding * 2))
    side = max(1, min(side, image_width, image_height))

    x1 = int(round(cx - (side / 2)))
    y1 = int(round(cy - (side / 2)))
    x1 = max(0, min(x1, image_width - side))
    y1 = max(0, min(y1, image_height - side))

    return x1, y1, x1 + side, y1 + side


def save_crop(image, box, output_path, size):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), crop)


def random_healthy_box(mask, crop_side, attempts=100):
    height, width = mask.shape[:2]
    if width < crop_side or height < crop_side:
        return None

    for _ in range(attempts):
        x1 = random.randint(0, width - crop_side)
        y1 = random.randint(0, height - crop_side)
        x2 = x1 + crop_side
        y2 = y1 + crop_side

        if cv2.countNonZero(mask[y1:y2, x1:x2]) == 0:
            return x1, y1, x2, y2

    return None


def extract_caries_and_healthy_crops(
    dataset_root,
    output_root,
    splits=("Train", "Supplemental content93"),
    padding=40,
    output_size=256,
    min_contour_area=50,
    healthy_per_caries=1,
    seed=42,
):
    random.seed(seed)

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    caries_dir = output_root / "caries"
    healthy_dir = output_root / "healthy"
    caries_dir.mkdir(parents=True, exist_ok=True)
    healthy_dir.mkdir(parents=True, exist_ok=True)

    caries_count = 0
    healthy_count = 0
    missing_masks = 0
    unreadable = 0

    for split in splits:
        image_dir = dataset_root / split / "images"
        mask_dir = dataset_root / split / "mask"

        if not image_dir.exists() or not mask_dir.exists():
            print(f"Skipping {split}: images or mask folder not found.")
            continue

        image_paths = find_image_files(image_dir)
        print(f"Processing {split}: found {len(image_paths)} images.")

        for image_path in image_paths:
            mask_path = matching_mask_path(image_path, mask_dir)
            if mask_path is None:
                missing_masks += 1
                print(f"  Missing mask for {image_path.name}")
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                unreadable += 1
                print(f"  Could not read {image_path.name} or {mask_path.name}")
                continue

            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # The provided pseudo masks use a dark label color. Any non-black
            # pixel belongs to the annotated caries region.
            _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_caries_boxes = []
            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                box = padded_square_box(x, y, w, h, image.shape[1], image.shape[0], padding)
                image_caries_boxes.append(box)

                output_path = caries_dir / f"{split.replace(' ', '_').lower()}_{image_path.stem}_{caries_count:04d}.jpg"
                save_crop(image, box, output_path, output_size)
                caries_count += 1

            if not image_caries_boxes:
                continue

            crop_side = max(32, int(np.median([box[2] - box[0] for box in image_caries_boxes])))
            crop_side = min(crop_side, image.shape[1], image.shape[0])
            target_healthy = len(image_caries_boxes) * healthy_per_caries

            for _ in range(target_healthy):
                box = random_healthy_box(binary_mask, crop_side)
                if box is None:
                    break

                output_path = healthy_dir / f"{split.replace(' ', '_').lower()}_{image_path.stem}_{healthy_count:04d}.jpg"
                save_crop(image, box, output_path, output_size)
                healthy_count += 1

    print("\nDataset crop complete.")
    print(f"  Caries crops:  {caries_count} -> {caries_dir}")
    print(f"  Healthy crops: {healthy_count} -> {healthy_dir}")
    print(f"  Missing masks: {missing_masks}")
    print(f"  Unreadable:    {unreadable}")

    return caries_count, healthy_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop caries and healthy regions from the dental X-ray dataset.")
    parser.add_argument(
        "--dataset-root",
        default="Childrens dental caries segmentation dataset",
        help="Root folder containing Train/Test/Supplemental content93 folders.",
    )
    parser.add_argument("--output-root", default="dataset", help="Output folder for caries/healthy crops.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Supplemental content93"],
        help="Dataset splits to crop. Example: --splits Train Test",
    )
    parser.add_argument("--padding", type=int, default=40, help="Pixels of padding around each caries mask.")
    parser.add_argument("--size", type=int, default=256, help="Output crop width and height.")
    parser.add_argument("--healthy-per-caries", type=int, default=1, help="Healthy crops to create per caries crop.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for healthy crop sampling.")
    args = parser.parse_args()

    extract_caries_and_healthy_crops(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        splits=tuple(args.splits),
        padding=args.padding,
        output_size=args.size,
        healthy_per_caries=args.healthy_per_caries,
        seed=args.seed,
    )
