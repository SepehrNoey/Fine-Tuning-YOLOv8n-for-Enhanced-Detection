# Fine-Tuning YOLOv8 for Enhanced Detection in Crowded Scenes

This project fine-tunes the YOLOv8n model on a custom dataset of crowded scenes with more cars and pedestrians. The goal was to improve the model's ability to detect and count vehicles and pedestrians in busy environments. After training for 100 epochs, the model showed some improvements over the pre-trained version in specific settings, though some misclassifications and overcounting still occur, requiring further training for more accuracy.


## Demo
Below is a comparison video between the pre-trained YOLOv8n model and the fine-tuned model on a custom dataset of crowded scenes:

- **Fine-Tuned YOLOv8n**: Detects more cars and pedestrians in crowded scenes.
- **Pre-Trained YOLOv8n**: Misses several objects in similar scenes.

#### Pedestrian Zones
|Fine-Tuned Model Output|Pre-trained Model Output|
|-----|----|
|<video src="https://github.com/user-attachments/assets/d5a87dde-a8d8-409a-ae0f-a3c77579c238">  | <video src="https://github.com/user-attachments/assets/14ca8d30-90cc-4ca1-a2b5-76a802f51cd4">|
|<video src="https://github.com/user-attachments/assets/0ed02d50-47aa-4f16-87a3-1339663defcf">  | <video src="https://github.com/user-attachments/assets/6bbbe0b6-d864-49a1-9e4b-50a9196768a2">|

#### Traffic Scenes
|Fine-Tuned Model Output|Pre-trained Model Output|
|-----|----|
|<video src="https://github.com/user-attachments/assets/dca43fc3-1296-4f6d-a0b1-8bcd0e5829b1">  | <video src="https://github.com/user-attachments/assets/a00cea11-64cc-4178-8596-b82c729fa727">|
|<video src="https://github.com/user-attachments/assets/d65c3017-af23-445f-a3bf-12be7ddbfcd2">  | <video src="https://github.com/user-attachments/assets/35719e19-47f0-4e80-94d7-163dfb5c6142">|
|<video src="https://github.com/user-attachments/assets/0722f36f-3e49-4760-aaec-6d0af848fb83">  | <video src="https://github.com/user-attachments/assets/58f43290-9269-4cd4-872a-50b56a7d1afe">|


## Dataset and Labeling
The custom dataset was created by gathering images of crowded scenes and labeling the following object classes using the [labelImg](https://github.com/tzutalin/labelImg) tool:

- **person**
- **bicycle**
- **car**
- **truck**

This step allowed for accurate fine-tuning of the YOLOv8n model to improve detection performance on these specific categories.

## Model Fine-Tuning
The pre-trained YOLOv8n model was fine-tuned on the custom dataset for **100 epochs**. This process improved its accuracy in crowded scenes, particularly in detecting pedestrians. As shown in the chart, the overall loss has reduced over the epochs and led to improved accuracy. Code of this part can be seen at `trainer.py`

![results](https://github.com/user-attachments/assets/cda4ec1e-10ad-4a2e-9e40-fc253c896977)


## Object Counting
After fine-tuning the model, some tools provided by the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library to count object in video sequences have been used. The code of this part can be seen at `counter.py`.

### Models Checkpoints
- **fine_tuned_yolov8n.pt**: This is the best checkpoint of the fine-tuned model. Weights and other arguments can be found at `runs/detect/train`
- **yolov8n.pt**: This is the base checkpoint used for fine-tuning on custom dataset, which is available at [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
