RDD2022 multicountry validation rerun

The recovered YOLOv8s checkpoint is from the Colab run s640.
The evaluation used the original 2,359-image validation split (5,480 boxes),
960 px, batch size 4, confidence 0.31 for precision/recall/F1, matching IoU 0.50,
NMS IoU 0.70, max detections 300, and no test-time augmentation.
The rerun used Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0,
Ultralytics 8.4.174, and OpenCV 4.11.0.86 on Apple M4 CPU.

In a Python environment, from this folder:
  python -m pip install torch==2.8.0 torchvision==0.23.0 opencv-python==4.11.0.86 ultralytics==8.4.174
  python evaluate.py data

The evaluation verifies the checkpoint, manifest, images, and labels by SHA-256.
The original Colab notebook is included as provenance. The data folder contains
all files needed for this validation rerun. Rerun results are written to
rerun_results.json. The old Word report describes an approximate F1 of 0.60
without a run ID or split, so its historical attribution remains unverified.
