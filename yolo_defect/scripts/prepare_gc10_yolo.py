from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_class_id(raw_name: str, folder_name: str) -> int:
    prefix_text = raw_name.split("_", 1)[0].strip() if raw_name else ""
    if prefix_text.isdigit():
        return int(prefix_text) - 1
    if folder_name.strip().isdigit():
        return int(folder_name.strip()) - 1
    raise ValueError(f"Unable to resolve class id from name={raw_name!r}, folder={folder_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GC10-DET Pascal VOC XML annotations into YOLO format."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/GC10-DET-meta"),
        help="GC10-DET source directory containing class folders and the 'lable' folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/gc10_yolo"),
        help="Output directory for YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating hard links.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_annotation(xml_path: Path) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    folder_name = root.findtext("folder", default="")
    width = int(root.findtext("size/width"))
    height = int(root.findtext("size/height"))

    objects: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip()
        if not class_name:
            continue

        class_id = resolve_class_id(class_name, folder_name)

        xmin = float(obj.findtext("bndbox/xmin"))
        ymin = float(obj.findtext("bndbox/ymin"))
        xmax = float(obj.findtext("bndbox/xmax"))
        ymax = float(obj.findtext("bndbox/ymax"))

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        objects.append((class_id, x_center, y_center, box_width, box_height))

    return filename, width, height, objects


def write_yolo_label(target_path: Path, objects: list[tuple[int, float, float, float, float]]) -> None:
    lines = [
        f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        for class_id, x_center, y_center, box_width, box_height in objects
    ]
    target_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def collect_images(source_dir: Path) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    for child in source_dir.iterdir():
        if not child.is_dir() or child.name == "lable":
            continue
        for image_path in child.iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths[image_path.name] = image_path
    return image_paths


def discover_class_names(xml_dir: Path) -> list[str]:
    class_names: dict[int, str] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        folder_name = root.findtext("folder", default="")
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", default="").strip()
            if not raw_name:
                continue
            class_index = resolve_class_id(raw_name, folder_name)
            _, _, label_text = raw_name.partition("_")
            fallback_label = f"class_{class_index + 1}"
            class_names[class_index] = label_text or class_names.get(class_index, fallback_label)
    return [class_names[idx] for idx in sorted(class_names)]


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    xml_dir = source_dir / "lable"
    output_dir = args.output.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {xml_dir}")

    image_paths = collect_images(source_dir)
    class_names = discover_class_names(xml_dir)
    if not class_names:
        raise RuntimeError("No classes discovered from XML annotations.")

    records: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    missing_images: list[str] = []
    class_counter: Counter[int] = Counter()

    for xml_path in sorted(xml_dir.glob("*.xml")):
        filename, _, _, objects = load_annotation(xml_path)
        image_path = image_paths.get(filename)
        if image_path is None:
            missing_images.append(filename)
            continue
        if not objects:
            continue
        records.append((image_path, objects))
        for class_id, *_ in objects:
            class_counter[class_id] += 1

    annotated_names = {image_path.name for image_path, _ in records}
    unannotated_images = sorted(name for name in image_paths if name not in annotated_names)

    if not records:
        raise RuntimeError("No valid image/annotation pairs found.")

    random.seed(args.seed)
    random.shuffle(records)
    val_count = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_count]
    train_records = records[val_count:]

    for split in ("train", "val"):
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    for split, split_records in (("train", train_records), ("val", val_records)):
        for image_path, objects in split_records:
            target_image = output_dir / "images" / split / image_path.name
            target_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy(image_path, target_image, args.copy_images)
            write_yolo_label(target_label, objects)

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    report_lines = [
        f"source: {source_dir}",
        f"output: {output_dir}",
        f"classes: {len(class_names)}",
        f"train_images: {len(train_records)}",
        f"val_images: {len(val_records)}",
        f"missing_images_for_xml: {len(missing_images)}",
        f"images_without_xml: {len(unannotated_images)}",
        "",
        "class_counts:",
    ]
    report_lines.extend(f"  {idx}: {class_names[idx]} -> {class_counter[idx]}" for idx in sorted(class_counter))
    if missing_images:
        report_lines.extend(["", "xml_without_matching_image:"])
        report_lines.extend(f"  {name}" for name in missing_images)
    if unannotated_images:
        report_lines.extend(["", "image_without_xml:"])
        report_lines.extend(f"  {name}" for name in unannotated_images)
    (output_dir / "conversion_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"Prepared YOLO dataset at: {output_dir}")
    print(f"Train images: {len(train_records)}")
    print(f"Val images: {len(val_records)}")
    print(f"Images without XML: {len(unannotated_images)}")
    print(f"XML without matching image: {len(missing_images)}")


if __name__ == "__main__":
    main()
