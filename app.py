
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 160
CONFIDENCE_THRESHOLD = 0.60

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


def format_class_name(name):
    name = name.replace("Tomato___", "")
    name = name.replace("_", " ")
    name = name.replace("Two spotted", "Two-spotted")
    return name


model = load_model()
class_names = load_classes()

st.title("🌿 PlantCare AI")
st.subheader("Tomato Leaf Disease Classifier")

st.write(
    "Upload a clear tomato leaf image to predict whether "
    "it is healthy or shows signs of disease."
)

st.info(
    "For best results, upload a well-lit, close-up photo "
    "of a single tomato leaf."
)

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Analyze Leaf", type="primary"):
            if image.width < 80 or image.height < 80:
                st.error(
                    "Image resolution is too low. "
                    "Please upload a clearer image."
                )
                st.stop()

            resized = image.resize((IMG_SIZE, IMG_SIZE))
            image_array = np.asarray(resized, dtype=np.float32)

            # Basic check for blank or nearly uniform images
            gray = np.mean(image_array, axis=2)

            if (
                float(np.std(image_array)) < 15
                or float(np.std(gray)) < 10
            ):
                st.error(
                    "This image appears blank or too uniform. "
                    "Please upload a clear tomato-leaf photo."
                )
                st.stop()

            # Keep the same input preprocessing as model training.
            # MobileNetV2 preprocessing is included inside the model.
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing leaf..."):
                probabilities = model.predict(
                    image_array, verbose=0
                )[0]

            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            top_indices = np.argsort(probabilities)[::-1][:3]

            predicted_class = format_class_name(
                class_names[best_idx]
            )

            if predicted_class.lower() == "healthy":
                st.success("🌿 Prediction: Healthy Tomato Leaf")
            else:
                st.success(f"🔎 Prediction: {predicted_class}")

            st.metric("Model confidence", f"{confidence:.1%}")

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    "The model has low confidence in this prediction. "
                    "Try a clearer image and verify the result."
                )

            st.write("### Top 3 predictions")

            for idx in top_indices:
                idx = int(idx)
                label = format_class_name(class_names[idx])
                score = float(probabilities[idx])

                st.write(f"**{label}** — {score:.1%}")
                st.progress(score)

            st.caption(
                "The model predicts among 10 tomato-leaf classes. "
                "It may misclassify unrelated images, and confidence "
                "does not confirm that an image contains a tomato leaf. "
                "Results are not a confirmed plant disease diagnosis."
            )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG or PNG file."
        )
