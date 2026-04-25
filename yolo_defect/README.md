# YOLO Industrial Defect Detection Coursework

## Project goal

Coursework topic:

`基于 YOLO 的工业表面缺陷检测研究——以 GC10-DET 数据集为例`

Current scope:

- Task type: object detection
- Framework: YOLOv5
- Dataset: GC10-DET
- Baseline model: `yolov5s`

## Current project status

Done:

- cloned YOLOv5 framework under `third_party/yolov5`
- collected GC10-DET source data under `datasets/GC10-DET-meta`
- converted Pascal VOC XML annotations into YOLO format
- generated train/val split and `data.yaml`

Key files:

- dataset config: `datasets/gc10_yolo/data.yaml`
- conversion script: `scripts/prepare_gc10_yolo.py`
- conversion report: `datasets/gc10_yolo/conversion_report.txt`

## Remaining work

1. Create Python environment and install dependencies.
2. Run baseline training with `yolov5s`.
3. Export training curves, metrics, and visual prediction samples.
4. Design at least one optimization experiment.
5. Write coursework report following the required template.
6. Package code and dataset links for submission.

## Recommended experiment path

### Stage 1: baseline

Use the original YOLOv5s model and train on GC10-DET.

Suggested first run:

```powershell
cd E:\Code\CV_EXP\CV_Final\yolo_defect\third_party\yolov5
python train.py --img 640 --batch 8 --epochs 100 --data ..\..\datasets\gc10_yolo\data.yaml --weights yolov5s.pt --name gc10_yolov5s_baseline
```

Expected outputs:

- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`
- `best.pt`
- `last.pt`

### Stage 2: improvement

Choose one realistic optimization direction:

- data augmentation adjustment
- lightweight comparison: `yolov5n` vs `yolov5s`
- input size comparison: `img 512` vs `img 640`
- training strategy comparison: different epochs / batch sizes
- module improvement if you still have time

For coursework delivery, the first three are safer than a deep architecture rewrite.

### Stage 3: inference examples

Run validation or detection on several images and keep:

- successful detection samples
- missed detection samples
- false detection samples

These are needed in the report analysis section.

## Submission checklist

You need to submit:

- printed report
- electronic report
- zipped code
- public dataset link

Do not put the raw dataset itself into GitHub unless the course explicitly requires it.
Prefer to submit:

- code
- scripts
- config files
- trained weights if size is acceptable
- dataset download link

## Suggested final package structure

```text
yolo_defect_submit/
├─ report/
│  ├─ 大作业报告.docx
│  └─ figures/
├─ code/
│  ├─ scripts/
│  ├─ third_party/
│  ├─ datasets/
│  │  └─ gc10_yolo/
│  └─ README.md
└─ dataset_links.txt
```

## Dataset links

- GC10-DET metadata repo: `https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets`
- GC10-DET original download page is documented in `datasets/GC10-DET-meta/README.md`

