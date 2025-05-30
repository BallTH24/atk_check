from google.colab import drive
import os
from PIL import Image
from roboflow import Roboflow


drive.mount('/content/drive')
rf = Roboflow(api_key="RJu6VkIYQ5QiQYCfwnvA")
project = rf.workspace("al-for-image-processing-v2").project("atk-v2")
model = project.version(1).model


folder_path = "/content/drive/MyDrive/atk_resultV2 (File responses)/ATK result (File responses)"
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path,filename)
    pred = model.predict(image_path, confidence=40, overlap=30)
    for each_pred in pred.predictions:
        if each_pred['confidence'] > 0.4 and each_pred['class'] == 'Pos':
        print(image_path)
        print(each_pred['confidence'],each_pred['class'])
        image = Image.open(image_path)
        display(image)