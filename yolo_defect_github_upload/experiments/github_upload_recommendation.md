# GitHub Upload Recommendation

## Should upload

- `README.md`
- `PACKAGE_README.md`
- `.gitignore`
- `scripts/`
- `figures/`
- `experiments/coursework_checklist.md`
- `experiments/results_summary.md`
- `experiments/upload_plan.md`
- `experiments/github_upload_recommendation.md`
- `experiments/weights/`
- `datasets/gc10_yolo/data.yaml`
- `datasets/gc10_yolo/conversion_report.txt`
- `datasets/GC10-DET-meta/README.md`
- `third_party/yolov5/models/common.py`
- `third_party/yolov5/models/yolov5s_se.yaml`
- `third_party/yolov5/models/yolov5s_cbam.yaml`
- `third_party/yolov5/utils/general.py`
- `third_party/yolov5/detect.py`
- `third_party/yolov5/val.py`
- `third_party/yolov5/runs/my_all_experiments/`
- `third_party/yolov5/runs/val/`
- `third_party/yolov5/runs/detect/`

## Optional upload

- a curated subset of `third_party/yolov5` files if you want the repo to stay smaller
- selected representative images from `runs/detect/` if you do not want full detection outputs online

## Should not upload

- raw dataset images under `datasets/GC10-DET-meta/1-10/`
- raw XML labels under `datasets/GC10-DET-meta/lable/`
- YOLO payload images and labels under `datasets/gc10_yolo/images/` and `datasets/gc10_yolo/labels/`
- pretrained checkpoint files `third_party/yolov5/yolov5s.pt` and `third_party/yolov5/yolov5n.pt`
- `third_party/yolov5/.git/`
- `datasets/GC10-DET-meta/.git/`
- cache or `__pycache__` files

