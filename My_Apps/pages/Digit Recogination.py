```python
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# 1. Cache and load trained model
@st.cache_resource
def load_digit_model():
    # Get the main My_Apps project folder
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Path to digit_model.h5
    model_path = BASE_DIR / "digit_model.h5"

    # Check if the model file exists
    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path}\n\n"
            "Please make sure 'digit_model.h5' is inside the main My_Apps folder."
        )
        st.stop()

    return tf.keras.models.load_model(model_path)


model = load_digit_model()


# 2. Image Preprocessing Helper Function
def preprocess_digit_image(img):
    """Preprocesses BGR image matching the training pipeline:

    Grayscale -> Adaptive Threshold -> Morph Open -> Square Pad -> Resize (32x32)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]

        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit_crop = thresh[y : y + h, x : x + w]

            max_dim = max(w, h)
            pad_size = max_dim + 20
            padded = np.zeros((pad_size, pad_size), dtype=np.uint8)

            start_x = (pad_size - w) // 2
            start_y = (pad_size - h) // 2

            padded[
                start_y : start_y + h,
                start_x : start_x + w
            ] = digit_crop

        else:
            padded = thresh
    else:
        padded = thresh

    resized = cv2.resize(
        padded,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype("float32") / 255.0
    input_data = np.expand_dims(normalized, axis=(0, -1))

    return resized, input_data


# 3. Updated Video Processing Class (VideoProcessorBase)
class RealTimeDigitRecognizer(VideoProcessorBase):

    def recv(self, frame):
        # Convert incoming WebRTC frame to BGR array
        img = frame.to_ndarray(format="bgr24")

        # Extract preprocessed array and normalized tensor
        resized, input_data = preprocess_digit_image(img)

        # Run model prediction
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_digit = np.argmax(prediction)
        confidence = prediction[predicted_digit] * 100

        # Overlay text prediction directly onto live webcam frame
        text = f"Digit: {predicted_digit} ({confidence:.1f}%)"

        cv2.putText(
            img,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        # Return updated VideoFrame object back to client stream
        return frame.from_ndarray(img, format="bgr24")


# 4. Streamlit Dashboard Layout
st.title("🔢 Real-Time Handwritten Digit Recognition")

st.write(
    "Choose between **Live Real-Time Camera Feed** or "
    "**Image File Upload** to recognize handwritten digits."
)

input_mode = st.radio(
    "Select Input Method:",
    ["Real-Time Live Camera 📹", "Upload Image File 📁"],
    horizontal=True,
)

st.markdown("---")


if input_mode == "Real-Time Live Camera 📹":

    st.subheader("Live Camera Feed")

    st.info(
        "Hold a handwritten digit up to your webcam. "
        "The model will continuously draw predictions over the video."
    )

    webrtc_streamer(
        key="digit-realtime",
        video_processor_factory=RealTimeDigitRecognizer,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
    )


else:

    st.subheader("Upload Digit Image")

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit picture...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Uploaded Image:**")
            st.image(
                img,
                use_container_width=True
            )

        resized, input_data = preprocess_digit_image(img)

        prediction = model.predict(
            input_data,
            verbose=0
        )[0]

        predicted_digit = np.argmax(prediction)
        confidence = prediction[predicted_digit] * 100

        with col2:
            st.write("**Recognition Result:**")

            st.metric(
                label="Predicted Digit",
                value=str(predicted_digit)
            )

            st.write(
                f"**Confidence Level:** `{confidence:.2f}%`"
            )

            st.image(
                resized,
                caption="Preprocessed Input to CNN (32x32)",
                width=160,
            )

            st.write(
                "**Probability breakdown across digits (0–9):**"
            )

            st.bar_chart(prediction)
```
