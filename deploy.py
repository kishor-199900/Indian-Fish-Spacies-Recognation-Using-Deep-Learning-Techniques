#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define the prediction function
def predict(image):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0

    # Make the prediction
    prediction = model.predict(image)
    label = np.argmax(prediction)

    return label

# Define the Streamlit app
st.title('Fish Recognation App')


uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Read the image file from the uploader
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    fish=['Catla','Rohu','Tilapia','Tuna']

    # Make the prediction
    label = predict(image)
    st.write(f'Prediction: {fish[label]}')
    




