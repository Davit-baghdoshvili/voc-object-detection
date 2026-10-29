# Technical Report: Pascal VOC Object Detection

## 1. Problem
Build a computer vision system that recognizes and localizes objects in images
using bounding boxes.

## 2. Data preparation
Pascal VOC 2012 provides XML annotations with object class names and bounding
box coordinates. These annotations were converted into YOLO format:
class_id, normalized_center_x, normalized_center_y, normalized_width,
normalized_height.

The training split contained 5,717 images and 15,774 object annotations.
The validation split contained 5,823 images and 15,787 object annotations.
Data checks found no invalid converted annotations. All 20 classes were represented.
Class frequencies were imbalanced.

## 3. Model and training
A pretrained YOLO11n detector was fine-tuned for 10 epochs using image size
640, batch size 16, and random seed 42 on a Tesla T4 GPU.

## 4. Baseline results
Precision = 0.765; Recall = 0.659; mAP@50 = 0.727; mAP@50–95 = 0.535.

Precision measures how many predicted detections are correct. Recall measures
how many ground-truth objects are detected. mAP@50 averages class-level
average precision at IoU threshold 0.50. mAP@50–95 averages over IoU thresholds
from 0.50 to 0.95 and penalizes less accurate box localization more strongly.

## 5. Experiment: image size
The same trained weights were evaluated at 512 px without retraining.

| Metric | 640 px baseline | 512 px |
|---|---:|---:|
| Precision | 0.765 | 0.7772 |
| Recall | 0.659 | 0.6414 |
| mAP@50 | 0.727 | 0.7183 |
| mAP@50–95 | 0.535 | 0.5300 |
| Inference time/image | 3.0 ms | 2.1 ms |

The smaller input size was faster in the reported validation run, while mAP
decreased slightly. Precision increased but recall decreased. This is a
speed/accuracy trade-off; results depend on hardware and evaluation conditions.

## 6. Strengths and limitations
YOLO11n is lightweight and supports fast inference. The system can detect
multiple VOC classes and display bounding boxes on uploaded images.
Limitations include class imbalance, missed or incorrect detections, sensitivity
to small objects and image conditions, and the lack of a separate held-out test set.

## 7. Future improvements
Use a separate test split, inspect per-class errors, test confidence thresholds,
compare image sizes and model variants, and evaluate performance on images
outside Pascal VOC.

## 8. AI tools disclosure
AI assistance was used to help draft code, documentation, and explanations.
The project owner should review, understand, and verify all code and results
before submission.
