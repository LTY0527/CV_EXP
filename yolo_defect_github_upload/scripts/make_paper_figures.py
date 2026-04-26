from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\Code\CV_EXP\CV_Final\yolo_defect")
YOLO_ROOT = ROOT / "third_party" / "yolov5"
OUT_DIR = ROOT / "figures"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_font(size: int):
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE_FONT = load_font(28)
LABEL_FONT = load_font(22)
SMALL_FONT = load_font(18)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    ratio = min(width / image.width, height / image.height)
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    offset = ((width - resized.width) // 2, (height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font, fill="black"):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = left + (right - left - tw) // 2
    y = top + (bottom - top - th) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_horizontal_triptych(items: list[tuple[str, Path]], out_path: Path, title: str) -> None:
    margin = 40
    gap = 24
    header_h = 70
    label_h = 40
    panel_w = 540
    panel_h = 360
    canvas_w = margin * 2 + panel_w * len(items) + gap * (len(items) - 1)
    canvas_h = margin * 2 + header_h + label_h + panel_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, title, (0, 10, canvas_w, header_h), TITLE_FONT)

    y0 = margin + header_h
    for idx, (label, path) in enumerate(items):
        x0 = margin + idx * (panel_w + gap)
        draw_centered(draw, label, (x0, y0, x0 + panel_w, y0 + label_h), LABEL_FONT)
        img = fit_image(Image.open(path), panel_w, panel_h)
        canvas.paste(img, (x0, y0 + label_h))
        draw.rectangle((x0, y0 + label_h, x0 + panel_w, y0 + label_h + panel_h), outline="black", width=2)
    canvas.save(out_path, quality=95)


def make_pr_confusion_grid(pr_items: list[tuple[str, Path]], cm_items: list[tuple[str, Path]], out_path: Path, title: str) -> None:
    margin = 36
    gap_x = 20
    gap_y = 30
    header_h = 60
    row_label_h = 44
    col_label_h = 44
    panel_w = 360
    panel_h = 280
    cols = 3
    rows = 2
    canvas_w = margin * 2 + cols * panel_w + (cols - 1) * gap_x
    canvas_h = margin * 2 + header_h + rows * (row_label_h + panel_h) + (rows - 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, title, (0, 8, canvas_w, header_h), TITLE_FONT)

    for row_idx, row_items in enumerate([pr_items, cm_items]):
        base_y = margin + header_h + row_idx * (row_label_h + panel_h + gap_y)
        row_title = "PR Curves" if row_idx == 0 else "Confusion Matrices"
        draw.text((margin, base_y + 4), row_title, font=LABEL_FONT, fill="black")
        for col_idx, (label, path) in enumerate(row_items):
            x0 = margin + col_idx * (panel_w + gap_x)
            draw_centered(draw, label, (x0, base_y, x0 + panel_w, base_y + col_label_h), SMALL_FONT)
            img = fit_image(Image.open(path), panel_w, panel_h)
            panel_top = base_y + row_label_h
            canvas.paste(img, (x0, panel_top))
            draw.rectangle((x0, panel_top, x0 + panel_w, panel_top + panel_h), outline="black", width=2)
    canvas.save(out_path, quality=95)


def make_single_row_grid(items: list[tuple[str, Path]], out_path: Path, title: str) -> None:
    margin = 36
    gap_x = 24
    header_h = 64
    label_h = 44
    panel_w = 360
    panel_h = 280
    canvas_w = margin * 2 + len(items) * panel_w + (len(items) - 1) * gap_x
    canvas_h = margin * 2 + header_h + label_h + panel_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered(draw, title, (0, 8, canvas_w, header_h), TITLE_FONT)

    y0 = margin + header_h
    for idx, (label, path) in enumerate(items):
        x0 = margin + idx * (panel_w + gap_x)
        draw_centered(draw, label, (x0, y0, x0 + panel_w, y0 + label_h), LABEL_FONT)
        img = fit_image(Image.open(path), panel_w, panel_h)
        canvas.paste(img, (x0, y0 + label_h))
        draw.rectangle((x0, y0 + label_h, x0 + panel_w, y0 + label_h + panel_h), outline="black", width=2)
    canvas.save(out_path, quality=95)


def main() -> None:
    ensure_dir(OUT_DIR)

    baseline_results = YOLO_ROOT / "runs" / "my_all_experiments" / "gc10_yolov5s_baseline_4090" / "results.png"
    mixup_results = YOLO_ROOT / "runs" / "my_all_experiments" / "exp_aug_mixup_640" / "results.png"
    se_results = YOLO_ROOT / "runs" / "my_all_experiments" / "exp_attention_se_640" / "results.png"

    make_horizontal_triptych(
        [
            ("Baseline (YOLOv5s-640)", baseline_results),
            ("MixUp", mixup_results),
            ("SE Attention", se_results),
        ],
        OUT_DIR / "figure3_training_curves_comparison.jpg",
        "Training Loss and mAP Curves Comparison",
    )

    make_pr_confusion_grid(
        [
            ("Baseline", YOLO_ROOT / "runs" / "val" / "val_baseline_local" / "PR_curve.png"),
            ("MixUp", YOLO_ROOT / "runs" / "val" / "val_mixup_local" / "PR_curve.png"),
            ("SE", YOLO_ROOT / "runs" / "val" / "val_se_local" / "PR_curve.png"),
        ],
        [
            ("Baseline", YOLO_ROOT / "runs" / "val" / "val_baseline_local" / "confusion_matrix.png"),
            ("MixUp", YOLO_ROOT / "runs" / "val" / "val_mixup_local" / "confusion_matrix.png"),
            ("SE", YOLO_ROOT / "runs" / "val" / "val_se_local" / "confusion_matrix.png"),
        ],
        OUT_DIR / "figure4_pr_confusion_comparison.jpg",
        "PR Curves and Confusion Matrices of Representative Models",
    )

    make_single_row_grid(
        [
            ("Baseline", YOLO_ROOT / "runs" / "val" / "val_baseline_local" / "PR_curve.png"),
            ("MixUp", YOLO_ROOT / "runs" / "val" / "val_mixup_local" / "PR_curve.png"),
            ("SE", YOLO_ROOT / "runs" / "val" / "val_se_local" / "PR_curve.png"),
        ],
        OUT_DIR / "figure4a_pr_curves_comparison.jpg",
        "PR Curves Comparison",
    )

    make_single_row_grid(
        [
            ("Baseline", YOLO_ROOT / "runs" / "val" / "val_baseline_local" / "confusion_matrix.png"),
            ("MixUp", YOLO_ROOT / "runs" / "val" / "val_mixup_local" / "confusion_matrix.png"),
            ("SE", YOLO_ROOT / "runs" / "val" / "val_se_local" / "confusion_matrix.png"),
        ],
        OUT_DIR / "figure4b_confusion_matrices_comparison.jpg",
        "Confusion Matrices Comparison",
    )

    sample_name = "img_01_425005700_00156.jpg"
    make_horizontal_triptych(
        [
            ("Baseline", YOLO_ROOT / "runs" / "detect" / "detect_baseline_local" / sample_name),
            ("MixUp", YOLO_ROOT / "runs" / "detect" / "detect_mixup_local" / sample_name),
            ("SE", YOLO_ROOT / "runs" / "detect" / "detect_se_local" / sample_name),
        ],
        OUT_DIR / "figure5_detection_visual_comparison.jpg",
        "Qualitative Detection Comparison on a Representative Defect Sample",
    )

    caption_text = """图3：不同方法下的训练 Loss 曲线与 mAP 曲线对比。

图4(a)：不同方法在验证集上的 PR 曲线对比。

图4(b)：不同方法在验证集上的混淆矩阵对比。

图5：baseline、MixUp 与 SE 模型的检测可视化对比。
"""
    (OUT_DIR / "figure_captions.txt").write_text(caption_text, encoding="utf-8")


if __name__ == "__main__":
    main()
