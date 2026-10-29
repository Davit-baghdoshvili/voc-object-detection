# Pascal VOC Object Detection with YOLO11n

## Project overview
This project fine-tunes a pretrained YOLO11n object detector on Pascal VOC 2012.
It detects and localizes objects using bounding boxes for 20 VOC classes.

## Dataset
Pascal VOC 2012 detection dataset.
- Training split: 5,717 images
- Validation split: 5,823 images
- Training annotations converted from Pascal VOC XML to YOLO format
- 20 object classes

The validation split was used to evaluate the trained model. It is not a separate
held-out test set.

## Model
- Architecture: YOLO11n
- Initialization: pretrained `yolo11n.pt`
- Training: 10 epochs, image size 640, batch size 16
- Training seed: 42
- Hardware: Google Colab Tesla T4 GPU

## Baseline validation metrics
- Precision: 0.765
- Recall: 0.659
- mAP@50: 0.727
- mAP@50–95: 0.535

## Image-size experiment
The same trained weights were evaluated at 512 pixels without retraining.
- Precision: 0.7772
- Recall: 0.6414
- mAP@50: 0.7183
- mAP@50–95: 0.5300
- Reported inference time: 2.1 ms/image at 512 px, versus 3.0 ms/image at 640 px
  in the baseline validation log on a Tesla T4. These timings are hardware-dependent.

## Run the demo
1. Install Python dependencies: `pip install -r requirements.txt`
2. Ensure `best.pt` is in the same folder as `app.py`.
3. Run: `streamlit run app.py`
4. Upload a JPG or PNG image and adjust the confidence threshold.

## Limitations
Performance varies by class, object size, image quality, and scene complexity.
The validation split is not an independent test set. The 512-pixel experiment
shows a speed/accuracy trade-off, not a universal improvement.

## Reproducibility
The trained weights are included as `best.pt`. For retraining, prepare Pascal VOC
2012 annotations in YOLO format and use the Ultralytics YOLO training API.
Exact package versions and the dataset preparation code should be recorded for
a fully reproducible environment.
