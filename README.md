# RS-YOLOv8

<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/input_video.gif" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/output_video.gif" width="49%" />
</p>

## Introduction
This project delivers an innovative, real-time solution for road segmentation within Autonomous Driving Assistant Systems (ADAS). It extends the capabilities of the RS-ADAS project by integrating the advanced object detection of YOLOv8 with an automated annotation process, addressing real-time processing challenges in ADAS applications.

## Project Objective
The primary goal is to develop a road segmentation model that operates efficiently in real-time environments. This is achieved by combining the object detection strengths of YOLOv8 with an automated annotation pipeline, leading to a robust and responsive ADAS component.

## Core Components
- **YOLOv8 Object Detection**: A state-of-the-art model used for identifying road areas in various settings.
- **Automated Annotation (autoannotate.py)**: A custom script leveraging Ultralytics's SAM large model for creating precise road segment annotations.
- **Ultralytics YOLOv8 Segmentation Model**: A model trained on the annotated data for accurate, real-time road segmentation.

## Detailed Workflow
<img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/yolo-segment.png" weight="20%">

- **Image Collection**: Gathering a diverse set of environmental images for model training.
- **Road Detection with YOLOv8**: Applying YOLOv8 for the initial detection of road areas in these images.
- **Automated Annotation Process**: Utilizing autoannotate.py to generate accurate annotations for the detected road segments.
- **Segmentation Model Training**: Using the annotated data to train the YOLOv8 segmentation model specifically for real-time application.
Deployment for Real-Time Inference: Implementing the trained model in ADAS for on-the-fly road segmentation.
