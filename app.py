
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plantcare_model.keras")

@st.cache_data
def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

st.title("🌿 PlantCare AI")
st.subheader("Tomato Leaf Disease Classifier")
st.write(
    "Upload a tomato leaf image to get an AI-generated "
    "prediction and confidence score."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=350)

    if st.button("Analyze Leaf", type="primary"):
        with st.spinner("Analyzing leaf..."):
            resized = image.resize((IMG_SIZE, IMG_SIZE))
            image_array = np.asarray(resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            probabilities = model.predict(image_array, verbose=0)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]

        best_idx = int(top_indices[0])
        confidence = float(probabilities[best_idx])

        st.success(f"Prediction: {class_names[best_idx]}")
        st.metric("Model confidence", f"{confidence:.1%}")

        st.write("### Top 3 predictions")
        for idx in top_indices:
            idx = int(idx)
            st.write(
                f"**{class_names[idx]}** — "
                f"{probabilities[idx]:.1%}"
            )
            st.progress(float(probabilities[idx]))

        st.caption(
            "This is an AI prediction, not a confirmed diagnosis. "
            "Image quality and visual similarity between diseases "
            "can affect results."
        )
