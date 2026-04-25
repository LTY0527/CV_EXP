# GitHub Upload Plan

## Recommended files to keep in the repository

- `README.md`
- `.gitignore`
- `scripts/prepare_gc10_yolo.py`
- `datasets/gc10_yolo/data.yaml`
- `datasets/gc10_yolo/conversion_report.txt`
- `datasets/GC10-DET-meta/README.md`
- `experiments/coursework_checklist.md`
- `third_party/yolov5/models/common.py`
- `third_party/yolov5/models/yolov5s_se.yaml`
- `third_party/yolov5/models/yolov5s_cbam.yaml`
- `third_party/yolov5/utils/general.py`
- `third_party/yolov5/val.py`
- `third_party/yolov5/detect.py`

## Files and folders to exclude

- raw dataset images and XML labels
- `datasets/gc10_yolo/images/`
- `datasets/gc10_yolo/labels/`
- `experiments/weights/`
- `third_party/yolov5/runs/`
- pretrained weights such as `yolov5s.pt` and `yolov5n.pt`
- nested `.git` directories from downloaded repos

## Why this split is appropriate

- The repository stays lightweight and public-friendly.
- The code changes and dataset preparation process remain reproducible.
- The report can cite the public dataset link instead of redistributing the dataset.
- Training logs and best weights can be attached separately if needed for submission.

## Suggested remote repository structure

```text
CV_EXP/
└─ yolo_defect/
   ├─ README.md
   ├─ .gitignore
   ├─ datasets/
   │  ├─ GC10-DET-meta/
   │  │  └─ README.md
   │  └─ gc10_yolo/
   │     ├─ data.yaml
   │     └─ conversion_report.txt
   ├─ experiments/
   │  ├─ coursework_checklist.md
   │  └─ upload_plan.md
   ├─ scripts/
   │  └─ prepare_gc10_yolo.py
   └─ third_party/
      └─ yolov5/
         ├─ detect.py
         ├─ val.py
         ├─ models/
         │  ├─ common.py
         │  ├─ yolov5s_se.yaml
         │  └─ yolov5s_cbam.yaml
         └─ utils/
            └─ general.py
```
