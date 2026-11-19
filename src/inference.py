import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from src.data_loader import IDX_TO_CHAR

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class ASLRecognizer:
    def __init__(self, model_path: str = "models/best_asl_cnn_multi.keras", labels_path: str = "models/class_names.txt"):
        self.model = tf.keras.models.load_model(model_path)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.landmarker = None
        if HAS_MEDIAPIPE:
            try:
                task_path = "models/hand_landmarker.task"
                if not os.path.exists(task_path):
                    os.makedirs("models", exist_ok=True)
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, task_path)
                base_options = mp_python.BaseOptions(model_asset_path=task_path)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=1,
                    min_hand_detection_confidence=0.35,
                )
                self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            except Exception:
                self.landmarker = None

        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
        else:
            num_out = self.model.output_shape[-1]
            if num_out == 26:
                self.class_names = [chr(ord("A") + i) for i in range(26)]
            else:
                self.class_names = [IDX_TO_CHAR[i] for i in range(len(IDX_TO_CHAR))]

    def _isolate_hand_on_clean_bg(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Crops tightly around the hand (via MediaPipe 21-point landmarks or YCrCb skin mask)
        and replaces cluttered backgrounds (curtains, clothes, shadows) with a clean
        uniform light-gray studio background matching Sign Language MNIST.
        """
        h, w, _ = image_bgr.shape
        x1, y1, x2, y2 = 0, 0, w, h
        found_box = False

        # 1. Locate exact hand bounding box using MediaPipe 21 landmarks
        if self.landmarker is not None:
            try:
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_img)
                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0]
                    xs = [int(lm.x * w) for lm in hand_lms]
                    ys = [int(lm.y * h) for lm in hand_lms]
                    x_min, x_max = max(0, min(xs)), min(w, max(xs))
                    y_min, y_max = max(0, min(ys)), min(h, max(ys))

                    bw, bh = x_max - x_min, y_max - y_min
                    side = int(max(bw, bh) * 1.25)
                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
                    x2, y2 = min(w, cx + side // 2), min(h, cy + side // 2)
                    found_box = True
            except Exception:
                found_box = False

        # 2. Fallback bounding box via YCrCb skin contour (with face suppression)
        if not found_box:
            ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
            mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
            gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_full, 1.1, 5, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces:
                mask[max(0, fy - int(fh * 0.25)) : min(h, fy + int(fh * 1.25)),
                     max(0, fx - int(fw * 0.15)) : min(w, fx + int(fw * 1.15))] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 0.015 * (h * w):
                    x, y, bw, bh = cv2.boundingRect(largest)
                    side = int(max(bw, bh) * 1.15)
                    cx, cy = x + bw // 2, y + bh // 2
                    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
                    x2, y2 = min(w, cx + side // 2), min(h, cy + side // 2)
                    found_box = True

        if not found_box:
            side = min(h, w)
            y1, x1 = (h - side) // 2, (w - side) // 2
            y2, x2 = y1 + side, x1 + side

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image_bgr

        # 3. Clean background inside the crop using GrabCut initialized with center hand rectangle
        ch, cw = crop.shape[:2]
        if ch >= 20 and cw >= 20:
            try:
                gc_mask = np.zeros((ch, cw), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                rect = (int(cw * 0.08), int(ch * 0.08), int(cw * 0.84), int(ch * 0.84))
                cv2.grabCut(crop, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

                # Smooth edges and composite onto uniform light-gray studio background (215)
                fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
                alpha = (fg_mask.astype(np.float32) / 255.0)[..., None]
                clean_bg = np.full_like(crop, 215, dtype=np.uint8)
                crop = (crop.astype(np.float32) * alpha + clean_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            except Exception:
                pass

        return crop

    def preprocess_image(self, image_bgr: np.ndarray, flip_horizontal: bool = False) -> np.ndarray:
        if flip_horizontal:
            image_bgr = cv2.flip(image_bgr, 1)

        roi = self._isolate_hand_on_clean_bg(image_bgr)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        resized = cv2.resize(enhanced, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        return normalized.reshape(1, 28, 28, 1)

    def predict_tensor(self, tensor_28x28: np.ndarray, top_k: int = 3):
        probs = self.model.predict(tensor_28x28, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            {"letter": self.class_names[int(i)], "confidence": float(probs[i])}
            for i in top_indices
        ]

    def predict_image(self, image_bgr: np.ndarray, top_k: int = 3, flip_horizontal: bool = False):
        tensor = self.preprocess_image(image_bgr, flip_horizontal=flip_horizontal)
        return self.predict_tensor(tensor, top_k=top_k), tensor.squeeze()