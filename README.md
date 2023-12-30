# RS-YOLOv8 - Road Segmentation with YOLOv8

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


## YOLOv8 Object Detection Architecture Overview
![U-Net Architecture with VGG Backbone](https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-SAM/blob/main/assets/yolo.jpg)
This image illustrates the detailed architecture of the YOLOv8 object detection model. It is a comprehensive schematic that outlines the flow from input image to detected objects.

### Backbone Architecture
The backbone is responsible for feature extraction and is constructed using a series of convolutional layers:

- **Pyramid Scaling Layers (P1 - P5)**: These layers form a feature pyramid that captures a wide range of object sizes and details.
- **CSPDarknet Layers**: Central to the backbone, they process the input images through a series of convolutions and shortcut connections.
- **C2F Blocks**: These are cross-stage partial blocks that enhance feature fusion by combining low and high-level information.
- **SPPF (Spatial Pyramid Pooling - Fast)**: This block pools features at different spatial scales to capture contextual information effectively.

### Head Architecture
The head is where the actual detection takes place and is comprised of:

- **YOLOv8 Detection Heads**: These are present for each scale (P3, P4, P5) and are responsible for predicting bounding boxes, objectness scores, and class probabilities.
- **Convolutional Layers**: They are used to process the feature maps and refine the detection results.
Upsampling Layers: These layers are utilized to merge feature maps from different levels of the backbone.
- **Loss Functions**: Includes Binary Cross-Entropy (BCE) for class prediction and Complete Intersection over Union (CIoU) loss for bounding box accuracy.
  
### Detection Process Details
- **Bottleneck and Concatenation**: Bottleneck layers followed by concatenation steps ensure rich feature maps that combine multiple levels of information.
- **Batch Normalization and SiLU Activation**: Included within convolutional blocks to stabilize learning and introduce non-linearities.
- **Detect Layers**: Located at strategic points in the architecture, they interpret the refined feature maps to make final object predictions.

### Dataset Composition

- **Training Set**: 5,000 images with corresponding segmentation masks.
- **Testing Set**: 1,00 images with associated masks for model accuracy evaluation.

### Preprocessing Techniques

- **Resizing**: Uniformly resized images and masks to 640 x 640 x 3 to standardize the input data.
- **Normalization**: Applied normalization to standardize pixel values across all images and eliminate outliers.

### Training Infrastructure

- Conducted on **Amazon SageMaker** with an NVIDIA Tesla T4 GPU (ml.g5.2xlarge instance).

### Training Hyperparameters

- **Epochs**: 100 epochs to balance learning and prevent overfitting.
- **Batch Size**: A batch size of 16, optimizing memory usage and model performance.
- **Learning Rate**: Set to 0.0001 for steady convergence without overshooting minima.
- **Custom Loss Function**: Binary Cross Entropy
- **Primary Metric**: Accuracy was used to gauge predictive performance.
- **Callbacks**: Early Stopping with a patience of 12 epochs and model checkpointing to save the best-performing model iteration.

<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/input_video.gif" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/output_video_box.gif" width="49%" />
</p>

## YOLOv8 Segmentation
- The Ultralytics' YOLOv8 segmentation model takes the images along with the anotated labels files(.txt) performs detailed segmentation, isolating the road in real time with high precision.
<p float="left">
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/output_video_box.gif" width="49%" style="margin-right: 2%;" />
  <img src="https://github.com/AnanthaPadmanaban-KrishnaKumar/RS-YOLOv8/blob/main/assets/output_video.gif" width="49%" />
</p>

## Major Achievements
- **Reduction in Inference Time**: One of the most noteworthy accomplishments of this project is the substantial reduction in inference time. By leveraging the efficiency of YOLOv8, the project successfully minimizes the time taken to process and segment road scenes. This improvement is crucial for real-time applications where decisions must be made swiftly and accurately.
- **Enhanced Accuracy and Precision**: The use of the YOLOv8 for the segmentation, with its detailed annotations, has enabled the model to segment roads with greater precision and accuracy. This enhancement is particularly beneficial in complex urban environments where distinguishing various elements accurately is vital for safe navigation.
- **Automated Annotation Process**: The introduction of an automated annotation process, reducing the reliance on manual annotation, has streamlined the model training phase. This not only saves significant time and effort but also reduces the likelihood of human error, leading to more consistent and reliable results.
- **Real-Time Processing Capability**: The ability of the system to process and analyze data in real-time is a critical requirement for ADAS, and this project successfully meets this demand. The real-time processing capability ensures that the ADAS can make prompt and informed decisions, a key factor in ensuring the safety of autonomous vehicles.

## Future Directions
As we continue to enhance our road segmentation model for Autonomous Driving Assistant Systems (ADAS), our ongoing efforts are directed towards optimizing the model for even faster processing speeds. A key part of this optimization includes integrating additional data sources to enhance the model's adaptability and accuracy.

In line with this objective, we are undertaking a new project utilizing the Cityscapes dataset. Cityscapes is a comprehensive dataset that provides a rich collection of images captured in urban environments, along with high-quality annotated labels for each image. These annotations cover various elements of urban street scenes, with a particular focus on roads. By leveraging this dataset, we aim to significantly improve the precision and accuracy of our road segmentation model. The Cityscapes dataset's detailed annotations will provide our model with deeper insights into complex urban landscapes, thus enabling more refined and accurate road detection capabilities in diverse environments. This integration will be instrumental in advancing the capabilities of our ADAS technologies, particularly in terms of enhanced reliability and efficiency in real-time applications.

## Conclusion
In conclusion, the YOLOv8 Segmentation project has set a new benchmark in the realm of ADAS. By successfully reducing inference time and enhancing the overall accuracy and efficiency of the segmentation process, this project not only addresses current technological challenges but also lays the groundwork for future innovations in autonomous vehicle navigation and traffic management. 
