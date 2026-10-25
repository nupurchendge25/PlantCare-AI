
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


model = load_model()
class_names = load_classes()

st.title("🌿 PlantCare AI")
st.subheader("Tomato Leaf Disease Classifier")

st.write(
    "Upload a clear tomato leaf image to predict its "
    "likely disease or healthy status."
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

            gray = np.mean(image_array, axis=2)
            color_variation = float(np.std(image_array))
            brightness_variation = float(np.std(gray))

            if (
                color_variation < 15
                or brightness_variation < 10
            ):
                st.error(
                    "The image appears blank or too uniform. "
                    "Please upload a clear tomato leaf photo."
                )
                st.stop()

            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing leaf..."):
                probabilities = model.predict(
                    image_array, verbose=0
                )[0]

            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            top_indices = np.argsort(probabilities)[::-1][:3]

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    "The model is uncertain about this image. "
                    "Please try a clearer tomato leaf photo."
                )
            else:
                st.success(
                    f"Most likely prediction: {class_names[best_idx]}"
                )

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
                "This model is trained on 10 tomato-leaf classes. "
                "It may incorrectly classify unrelated images, "
                "because confidence does not prove that an image "
                "contains a tomato leaf. Predictions are not a "
                "confirmed plant disease diagnosis."
            )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG or PNG file."
        )
