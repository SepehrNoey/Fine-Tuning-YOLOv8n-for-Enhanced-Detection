# Fine-Tuning YOLOv8 for Enhanced Detection in Crowded Scenes

This project involves fine-tuning the YOLOv8n model on a custom dataset of images featuring crowded scenes with more cars and pedestrians. The aim was to assess the ability to train a pre-trained model further for specific tasks, such as improving its ability to detect and count vehicles and pedestrians accurately in such environments. By gathering a targeted dataset and training the model for 100 epochs, the results show significant improvements in object detection compared to the pre-trained model, particularly in busy scenes. However, some misclassifications or overcounting of one object occur and the model must be trained for more accurate results and further academic purposes.

## Demo
Below is a comparison video between the pre-trained YOLOv8n model and the fine-tuned model on a custom dataset of crowded scenes:

- **Fine-Tuned YOLOv8n**: Detects more cars and pedestrians in crowded scenes.
- **Pre-Trained YOLOv8n**: Misses several objects in similar scenes.

https://private-user-images.githubusercontent.com/77605118/367538454-0e86ffbb-c58d-4ab3-a196-653a1b2dd929.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjYzNDExOTksIm5iZiI6MTcyNjM0MDg5OSwicGF0aCI6Ii83NzYwNTExOC8zNjc1Mzg0NTQtMGU4NmZmYmItYzU4ZC00YWIzLWExOTYtNjUzYTFiMmRkOTI5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MTQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTE0VDE5MDgxOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJmYzY3MDk4YmJmMmFlZWY0ZjBhZGQ5YTcwODQ0M2QxNWJmNGNmNjYwN2U5MGM3OTdmODQ5YWFhOGE3YWJlYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.b7j7Xr74ljYHeBQGUOGdqzBW2WHmUXJkTLvGJKPAbx8

## Dataset and Labeling
The custom dataset was created by gathering images of crowded scenes and labeling the following object classes using the [labelImg](https://github.com/tzutalin/labelImg) tool:

- **person**
- **bicycle**
- **car**
- **truck**

This step allowed for accurate fine-tuning of the YOLOv8n model to improve detection performance on these specific categories.

## Model Fine-Tuning
The pre-trained YOLOv8n model was fine-tuned on the custom dataset for **100 epochs**. This process significantly improved its accuracy in crowded scenes, particularly in detecting cars and pedestrians.

![Training Chart](training_chart_image_link_here)

The training chart shows:

- **Loss reduction over epochs**: Clear reduction in both classification and bounding box regression loss.
- **Accuracy improvement**: The model's accuracy improves steadily during training, particularly for the car and pedestrian categories.

## Object Counting
After fine-tuning, I used the tools provided by the [Ultralytics YOLO](https://github.com/ultralytics/yolov8) library to compare object counts in video sequences. The fine-tuned model was tested on crowded scenes and showed significant improvements in detecting and counting cars and people.

### Comparison Results

| Model                 | Total Cars Counted | Total People Counted |
|-----------------------|--------------------|----------------------|
| **Pre-trained YOLOv8n** | X cars             | X people             |
| **Fine-tuned YOLOv8n**  | Y cars             | Y people             |

## Conclusion
The fine-tuned YOLOv8n model performs much better in crowded scenes, accurately detecting and counting objects like cars and pedestrians compared to the pre-trained model. This project demonstrates the impact of custom training on real-world scenes and the potential improvements in object detection accuracy through targeted fine-tuning.

## How to Run

1. Clone this repository.
2. Install the necessary dependencies using:
   ```bash
   pip install -r requirements.txt
