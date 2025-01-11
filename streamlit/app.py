import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define paths dynamically for hosted environments
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'strawberry_disease_model.h5')
HEADER_IMAGE_PATH = os.path.join(BASE_DIR, 'header_image.png')

# Cache the model to avoid reloading it on every execution
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure it is included in the deployment.")

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
if os.path.exists(HEADER_IMAGE_PATH):
    st.image(HEADER_IMAGE_PATH, use_column_width=True)  # Add the header image
st.markdown("<h2>Strawberry Disease Classification App 🍓</h2> <br>", unsafe_allow_html=True)
st.write("Upload an image of a strawberry leaf to detect diseases.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Run prediction
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Create a two-column layout
        col1, col2 = st.columns([1, 1])  # Adjust the ratio if needed

        # Display the uploaded image in the first column
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display prediction results in the second column
        with col2:
            st.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                    <h4 style="color: #ff4b4b;">Predicted Disease:</h4>
                    <h3 style="color: #333;">{disease_labels[predicted_class]}</h3>
                    <h4 style="color: #ff4b4b;">Confidence:</h4>
                    <h3 style="color: #333;">{confidence:.2%}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Show confidence scores for all classes below the main result
        st.write("### Confidence Scores:")
        for i, score in enumerate(predictions[0]):
            st.write(f"{disease_labels[i]}: {score:.2%}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by [TechSquad].")
