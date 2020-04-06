import os
import cv2
from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #gives a path of current working directory
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            print(label, path)

            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # y_labels.append(label)
            # x_train.append(path)
            pil_image = Image.open(path).convert("L")   # grey scale
            image_array = np.array(pil_image, "uint8")  # convert into array
            print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi = image_array[y: y+h, x: x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)


