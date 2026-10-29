import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="VOC Object Detector", layout="wide")
st.title("Computer Vision: Object Detection")
st.write("YOLO11n fine-tuned on Pascal VOC 2012.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )
    annotated = results[0].plot()
    annotated = annotated[:, :, ::-1]

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Predictions")
        st.image(annotated, use_container_width=True)

    st.subheader("Detected objects")
    if len(results[0].boxes) == 0:
        st.write("No objects detected at this confidence threshold.")
    else:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            st.write(f"{model.names[class_id]} — {score:.1%}")
