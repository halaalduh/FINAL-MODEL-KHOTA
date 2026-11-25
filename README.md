# Khota

Khota v2 looks for curb ramp issues and surface damage in sidewalk photos. For each detection, it returns a bounding box, a confidence score and a severity rating (Low–Medium or High).

## Run the model

Use Python 3.11 and install PyTorch for your platform as described in the [installation guide](inference/INTEGRATION_GUIDE.md#3-requirements-and-installation). With Git LFS installed, run these commands from the repository root. Replace the example path with your own photo.

```bash
git lfs pull --include="inference/weights/*"
python -m pip install -r inference/requirements.txt
python inference/predict.py "/path/to/sidewalk.jpg"
```

The last command prints JSON. See the [model weights](inference/weights/) and an [example response](inference/examples/sample_response.json).

## Results

| v2 metric | Validation (104 photos) | Test (103 photos) |
| --- | ---: | ---: |
| Detection mAP@0.5 | 0.446 | 0.375 |
| Severity Macro-F1 on labelled defects | 0.788 | 0.761 |

The [results folder](results/) contains the measurements, plots and example detections.

## Repository guide

- [Data](data/) contains the annotations, dataset splits and instructions for preparing the training photos.
- [Detection](detection/) and [severity](severity/) contain the training scripts and settings. [Evaluation](evaluation/scripts/) contains the scoring scripts.
- The [RDD2022 experiment](notebooks/rdd2022/) and [archived v1 model](archive/v1/) are available for reference.
- The [project report](docs/REPORT.md) covers the experiments, data source and use of AI. See also the [PDF report](docs/Khota_Project_Report.pdf) and [licenses](docs/LICENSES.md).
