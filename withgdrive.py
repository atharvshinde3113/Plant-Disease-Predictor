import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload

# Set up working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Google Drive API setup (using a service account)
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'path_to_your_service_account.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Function to download images from Google Drive folder
def get_images_from_drive(folder_id):
    results = service.files().list(q=f"'{folder_id}' in parents",
                                   pageSize=10,
                                   fields="files(id, name)").execute()
    items = results.get('files', [])
    images = []
    
    if not items:
        st.write('No files found in the Google Drive folder.')
    else:
        for item in items:
            file_id = item['id']
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            fh.seek(0)
            img = Image.open(fh)
            images.append(img)
    return images


# Streamlit App
st.title('Leaf Disease Classifier')

# Option to upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Option to input a Google Drive folder ID
folder_id = st.text_input("Or enter a Google Drive folder ID:")

if uploaded_image is not None:
    # Process and classify the uploaded image
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

elif folder_id:
    # Process images from Google Drive folder
    if st.button('Fetch Images from Google Drive'):
        images = get_images_from_drive(folder_id)
        
        if images:
            for i, image in enumerate(images):
                st.image(image, caption=f"Image {i+1}")
                prediction = predict_image_class(model, image, class_indices)
                st.write(f'Prediction for Image {i+1}: {prediction}')
        else:
            st.write("No images found in the folder.")
