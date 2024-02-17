from ultralytics import YOLO
import numpy as np
import cv2
import streamlit as st
from PIL import Image

model_path=r"C:\Users\91808\Downloads\yolov8\best.pt"

image=r"C:\Users\91808\Downloads\yolov8\test_images\10_jpg.rf.26a9fccd4568b97f966399936250762a.jpg"

# img=cv2.imread(image)

model=YOLO(model_path)

st.title("Insert your image ")

image=st.file_uploader("upload image",type=["png","jpg","jpeg","gif"])

if image:

    image=Image.open(image)

    st.image(image=image)

    result=model(image)

    names=result[0].names

    probability=result[0].probs.data.numpy()

    prediction=np.argmax(probability)

    st.write(names)

    st.write(prediction)
    
    if prediction==0:
        st.write("This is Monkeypox ")
    elif prediction==1:
        st.write("This is Normal")
    elif prediction==2:
        st.write("This is Others")