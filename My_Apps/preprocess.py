import os
import glob
import cv2
import numpy as np

# Define directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (32, 32)


def preprocess_image(image_path):
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Warning: Unable to read image at {image_path}")
        return None

    # 2. Gaussian blur to remove high-frequency noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # 3. Adaptive thresholding to convert to binary (digit white, background black)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # 4. Morphological opening to eliminate isolated pixel noise while keeping loops open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5. Locate contours and crop around the digit
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Filter out negligible tiny noise contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            digit_crop = cleaned[y : y + h, x : x + w]

            # Square pad the cropped image to preserve aspect ratio
            max_dim = max(w, h)
            pad_size = max_dim + 20
            padded = np.zeros((pad_size, pad_size), dtype=np.uint8)

            start_x = (pad_size - w) // 2
            start_y = (pad_size - h) // 2
            padded[start_y : start_y + h, start_x : start_x + w] = digit_crop
        else:
            padded = cleaned
    else:
        padded = cleaned

    # 6. Resize to 32x32 target input size using area interpolation
    resized = cv2.resize(padded, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return resized


def process_all_data():
    if not os.path.exists(RAW_DIR):
        print(
            f"❌ Directory '{RAW_DIR}' not found. Please ensure raw dataset is uploaded."
        )
        return

    # Process each class subfolder (0 to 9) if present, or general image files
    image_paths = glob.glob(
        os.path.join(RAW_DIR, "**", "*.[jJ][pP][gG]"), recursive=True
    ) + glob.glob(
        os.path.join(RAW_DIR, "**", "*.[pP][nN][gG]"), recursive=True
    )

    if not image_paths:
        print(f"⚠️ No raw images found inside '{RAW_DIR}'.")
        return

    print(f"🔍 Found {len(image_paths)} images to preprocess...")

    processed_count = 0
    for img_path in image_paths:
        # Keep subfolder structures if present (e.g. data/raw/7/img1.jpg -> data/processed/7/img1.png)
        relative_path = os.path.relpath(img_path, RAW_DIR)
        save_path = os.path.join(
            PROCESSED_DIR, os.path.splitext(relative_path)[0] + ".png"
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        processed_img = preprocess_image(img_path)
        if processed_img is not None:
            cv2.imwrite(save_path, processed_img)
            processed_count += 1

    print(
        f"✅ Preprocessing complete! Successfully saved {processed_count} images to '{PROCESSED_DIR}'."
    )


if __name__ == "__main__":
    process_all_data()