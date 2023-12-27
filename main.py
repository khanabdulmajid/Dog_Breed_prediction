import numpy as np
# import pandas as pd
import tensorflow as tf
import streamlit as st
import cv2

model = tf.keras.models.load_model(r"C:\Users\abdul\Downloads\dogbreed.h5")

CLASS_NAMES = ["scottish deerhound","maltese dog","bernese mountain dog"]

st.title("DOG Breed Prediction")
st.markdown("Upload an image of the Dog")
dog_image = st.file_uploader("Choose an image ...",type="png")
submit = st.button("Predict")

if submit:
    if dog_image is not None:
        #  convert file to open cv image
        file_bytes = np.asarray(bytearray(dog_image.read()),dtype='uint8')
        opencv_image = cv2.imdecode(file_bytes, 1)

        #display the image

        st.image(opencv_image,channels="BGR")
        opencv_image=cv2.resize(opencv_image,(224,224))
        opencv_image.shape=(1,224,224,3)
        y_pred=model.predict(opencv_image)

        res = CLASS_NAMES[np.argmax(y_pred)].upper()
        # st.markdown("{:green[res]} ")

        st.markdown(f'<p style="color:#33ff33;text-align: center;font-size:40px;border-radius:2%;">{res}</p>',
                    unsafe_allow_html=True)

    else:
        st.markdown(f'<p style="color:red;text-align: center;font-size:40px;border-radius:2%;">oops!... Invalid Input</p>',
                    unsafe_allow_html=True)

