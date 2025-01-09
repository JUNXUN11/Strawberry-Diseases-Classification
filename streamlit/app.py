import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
MODEL_PATH = 'strawberry_disease_model.h5'

@st.cache_resource  # Cache the model to avoid reloading it on each run
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Define disease labels
disease_labels = {
    0: "Angular Leafspot",
    1: "Anthracnose Fruit Rot",
    2: "Blossom Blight",
    3: "Gray Mold",
    4: "Leaf Spot",
    5: "Powdery Mildew Fruit",
    6: "Powdery Mildew Leaf"
}

# Preprocess the image for prediction
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize image to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit App UI
st.title("Strawberry Disease Detection App 🍓")
st.write("Upload an image of a strawberry leaf to detect diseases.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Run prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Display results
    st.write(f"**Predicted Disease:** {disease_labels[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2%}")

    # Show confidence scores for all classes
    st.write("### Confidence Scores:")
    for i, score in enumerate(predictions[0]):
        st.write(f"{disease_labels[i]}: {score:.2%}")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by [TechSquad].")