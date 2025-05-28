import os
from PIL import Image
from roboflow import Roboflow
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient


# Connect to Roboflow
rf = Roboflow(api_key="RJu6VkIYQ5QiQYCfwnvA")
project = rf.workspace("al-for-image-processing-v2").project("atk-v2")
model = project.version(1).model

# Folder containing ATK result images
folder_path = "./atk_check/ATK_result"

# Loop through each image file
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        pred = model.predict(image_path, confidence=40, overlap=30)
        predictions = pred.json()['predictions']
        for each_pred in predictions:
            if each_pred['confidence'] > 0.4 and each_pred['class'] == 'Pos':
                print(f"🧪 Positive detected in: {filename}")
                print(f"Confidence: {each_pred['confidence']:.2f} | Class: {each_pred['class']}")
                image = Image.open(image_path)

                # แสดงผลรูปภาพด้วย matplotlib
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"{filename} - Positive Detected")
                plt.show()
