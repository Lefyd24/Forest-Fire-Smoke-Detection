import PIL
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import pandas as pd
import yolov7
import tempfile
#sys.path.append("yolov7")

model_yolov8 = "models/yolov8/weights/fire_model.pt"
model_yolov7 = "models/yolov7/runs/train/exp/weights/best.pt"

st.set_page_config(
    page_title="Forest Fire and Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    model_selection = st.selectbox("Choose a model:", ["YOLOv8", "YOLOv7"]) #  
    # Model selection dropdown
    st.header("Image Config")
    uploaded_file = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
   
    #st.header("Image/Video Config")
    #uploaded_file = st.file_uploader(
    #    "Upload an image or video...", type=("jpg", "jpeg", "png", "bmp", "webp", "mp4")
    #)x

    confidence = float(st.slider("Select Model Confidence", 15, 100, 20)) / 100

st.title("Fire and Smoke Detection using YOLOv8 & YOLOv7")
st.caption("a Project for MSc in Business Analytics (AUEB) - Machine Learning and Content Analytics 2023")
with st.expander("Model Characteristics"):
    chars_alt = pd.DataFrame({
        "Param": ["Dataset", "Images", "Epochs", "IMG_SIZE", "BATCH_SIZE", "LR"],
        "Value": ["https://universe.roboflow.com/kirzone/fire-iejes/dataset/2", 1706, 30, 640, 20, 0.01]
    })
    st._legacy_table(chars_alt)

st.caption(
    'Upload a photo/video and then click the "Detect Objects" button to view the results.'
)

col1, col2 = st.columns(2)

with col1:
    if uploaded_file:
        if uploaded_file.type in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
            uploaded_image = PIL.Image.open(uploaded_file)
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        else:
            st.video(uploaded_file)

@st.cache_resource
def load_yolov8_model():
    return YOLO(model_yolov8)

@st.cache_resource
def load_yolov7_model():
    return yolov7.load(model_yolov7)

# Load model based on user selection
if model_selection == "YOLOv8":
    try:

        model = load_yolov8_model()
    except Exception as ex:
        st.error(f"Unable to load YOLOv8 model. Check the specified path: {model_yolov8}")
        st.error(ex)
elif model_selection == "YOLOv7":
    try:
        model = load_yolov7_model()
    except Exception as ex:
        st.error(f"Unable to load YOLOv7 model. Check the specified path: {model_yolov7}")
        st.error(ex)

def process_image_detections(res, col2, model_selection):
    if model_selection == "YOLOv8":
        print(res[0].speed)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        col2.image(res_plotted, caption="Detected Image", use_column_width=True)

        fire_count = 0
        smoke_count = 0
        fire_confidences = []
        smoke_confidences = []

        with st.expander("Detection Results"):
            for box in boxes:
                label = model.names[int(box.cls)]
                confidence = box.conf.item()
                if label == "fire":
                    fire_count += 1
                    fire_confidences.append(confidence)
                    st.markdown(f"<span style='color:red'>{label.capitalize()}</span> - Confidence: {confidence:.2f} - Coordinates: {box.xywh}", unsafe_allow_html=True)
                elif label == "smoke":
                    smoke_count += 1
                    smoke_confidences.append(confidence)
                    st.markdown(f"<span style='color:blue'>{label.capitalize()}</span> - Confidence: {confidence:.2f} - Coordinates: {box.xywh}", unsafe_allow_html=True)
            st.write(f"Total Fires Detected: {fire_count}")
            st.write(f"Total Smokes Detected: {smoke_count}")
            st.markdown(f"<span style='color:green'>Preprocess Time:</span> {res[0].speed.get('preprocess'):.2f} ms<br><span style='color:green'>Inference Time:</span> {res[0].speed.get('inference'):.2f} ms<br><span style='color:green'>Postprocess Time:</span> {res[0].speed.get('postprocess'):.2f} ms", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Average Confidence per Class"):
                labels = ["Fire", "Smoke"]
                avg_confidences = [np.mean(fire_confidences) if fire_confidences else 0, 
                                np.mean(smoke_confidences) if smoke_confidences else 0]
                plt.figure(figsize=(8, 5))
                plt.bar(labels, avg_confidences, color=['#ff9999','#66b2ff'])
                plt.title("Average Confidence per Class")
                plt.ylabel("Average Confidence")
                plt.ylim(0, 1)  # Y-axis between 0 and 1 for clarity
                col1.pyplot(plt)

        with col2:
            with st.expander("Class Distribution"):
                labels = ["Fire", "Smoke"]
                sizes = [fire_count, smoke_count]
                colors = ['#ff9999','#66b2ff']
                plt.figure(figsize=(8, 8))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title("Distribution of Detected Classes")
                col2.pyplot(plt)
    else:
        col2.image(res, caption="Detected Image", use_column_width=True)


def process_video(uploaded_video, model, confidence):
    video_file = "temp_video.mp4"
    with open(video_file, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(
        temp_file.name, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height)
    )

    # Adding a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()  # Placeholder for the progress text

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=confidence)
        annotated_frame = res[0].plot()[:, :, ::-1]
        out.write(annotated_frame)

        # Updating the progress bar and text
        current_progress = int((i + 1) / total_frames * 100)
        progress_bar.progress(current_progress)
        progress_text.text(f"Progress: {current_progress}%")

    cap.release()
    out.release()
    # Delete temp video
    os.remove(video_file)

    return temp_file.name


if st.sidebar.button("Detect Objects"):
    if uploaded_file.type in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        # Save the uploaded file to a temporary location
        temp_file = "temp_uploaded_file.jpg"
        with open(temp_file, "wb") as f:    
            f.write(uploaded_file.getvalue())

        if model_selection == "YOLOv8":
            res = model.predict(uploaded_image, conf=confidence)
            process_image_detections(res, col2, model_selection)

        else:
            model.conf = confidence
            res = model(uploaded_image)
            
            with col2:
                predicted_image = res.render()
                st.image(predicted_image, caption="Uploaded Image", use_column_width=True)
        
    else:
        if model_selection == "YOLOv8":
            output_video = process_video(uploaded_file, model, confidence)
        else:
            pass
        with col2:
            if model_selection == "YOLOv8":
                #video_bytes = open(output_video, "rb").read()
                # Download the video instead of rendering it
                with open(output_video, "rb") as video_file:
                    video_data = video_file.read()
                st.download_button(label="Download Processed Video", data=video_data, file_name="processed_video.mp4",
                                    mime="video/mp4", use_container_width=True)

            else:
                with open(processed_video_path, "rb") as f:
                    st.video(f.read())

