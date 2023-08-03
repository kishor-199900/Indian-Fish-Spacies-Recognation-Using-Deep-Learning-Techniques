#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import base64



# # Load the background image using the 'Image' function
# bg_image = Image.open("fish-market-7519944.jpg")

# # Set the background image using the 'st.image' function
# st.image(bg_image, use_column_width=True)

# Define the Streamlit app
st.title('Fish Recognition App')

# Load the background image using the 'Image' function
bg_image = Image.open("fish-market-7519944.jpg")


# Set the background image
bg_image_url = 'https://www.shutterstock.com/image-photo/seafood-fish-market-260nw-275854940.jpg'
page_bg = '''
<style>
body {
background-image: url("%s");
background-size: contain;
}
</style>
''' % bg_image_url

# Display the background image
st.markdown(page_bg, unsafe_allow_html=True)



# Define the fish classes
fish = ['Catla','Mrigal', 'Rohu', 'Tilapia', 'Tuna']

# Define a function to make predictions
def predict(image):
    # Preprocess the image
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction using the model
    prediction = model.predict(image)

    # Return the predicted class
    return np.argmax(prediction)

# Load the machine learning model
model = tf.keras.models.load_model('model.h5')

# Add a sidebar with radio buttons to select the input option
option = st.sidebar.radio('Select Input Option', ('File Uploader', 'Live Camera'))

# If the user selects 'File Uploader'
if option == 'File Uploader':
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the image file from the uploader
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction
        label = predict(image)
        st.write(f'Prediction: {fish[label]}',size=10,color='r')
        st.write('<span style="font-size:36px; color:red">Hello World!</span>', unsafe_allow_html=True)

# If the user selects 'Live Camera'
elif option == 'Live Camera':
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Capture a single frame from the camera
    ret, frame = cap.read()

    # Make a prediction if a frame is captured
    if ret:
        # Display the captured frame
        st.image(frame, caption='Live Camera Feed', use_column_width=True)

        # Make a prediction
        label = predict(frame)
        st.write(f'Prediction: {fish[label]}',size=10,color='r')
        st.write('<span style="font-size:36px; color:red">Hello World!</span>', unsafe_allow_html=True)

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




