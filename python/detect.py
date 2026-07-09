"""
DEEPSHIELD AI Detection Server — Hybrid Approach
===================================================
Uses TWO models for best accuracy:

  • IMAGES → HuggingFace  umm-maybe/AI-image-detector  (ViT, 2-class)
  • VIDEOS → Local        deepfake_detection_model.h5   (Xception+LSTM, 2-class)

Endpoints:
  POST /detect        — classify an image
  POST /detect-video  — classify a video
  GET  /health        — server status

CLI:
  python detect.py <image_path>
  python detect.py <video_path> --video
  python detect.py --server
"""

import sys
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# Constants
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(SCRIPT_DIR, 'deepfake_detection_model.h5')
HF_MODEL_NAME = "umm-maybe/AI-image-detector"

IMG_SIZE = 128          # for the H5 video model
SEQUENCE_LENGTH = 10    # frames per video sequence

MESSAGES = {
    'AI-GENERATED': 'This media appears to be AI-generated (created by AI tools such as Midjourney, DALL-E, or Stable Diffusion).',
    'AI-EDITED': 'This media appears to have been edited or manipulated using AI tools.',
    'REAL': 'This media appears to be authentic with no AI manipulation detected.',
}


# ============================================================
# Image Model — HuggingFace  (lazy loaded)
# ============================================================

_image_classifier = None


def get_image_classifier():
    """Load HuggingFace image classification pipeline (lazy, cached)."""
    global _image_classifier
    if _image_classifier is not None:
        return _image_classifier

    from transformers import pipeline
    print(f"[DEEPSHIELD-PY] Loading image model: {HF_MODEL_NAME}")
    _image_classifier = pipeline("image-classification", model=HF_MODEL_NAME)
    print("[DEEPSHIELD-PY] Image model loaded!")
    return _image_classifier


# ============================================================
# Video Model — Local H5  (lazy loaded)
# ============================================================

_video_model = None


def build_video_model():
    """Reconstruct Xception+LSTM architecture & load saved weights."""
    import tensorflow as tf
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.layers import (
        Input, TimeDistributed, Flatten, Dropout, LSTM, Dense
    )
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(
            Xception(include_top=False, weights=None,
                     input_shape=(IMG_SIZE, IMG_SIZE, 3))
        ),
        TimeDistributed(Flatten()),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.load_weights(H5_MODEL_PATH)
    return model


def get_video_model():
    """Load the video model (lazy, cached)."""
    global _video_model
    if _video_model is not None:
        return _video_model

    if not os.path.exists(H5_MODEL_PATH):
        raise FileNotFoundError(
            f"Video model not found: {H5_MODEL_PATH}\n"
            f"Place 'deepfake_detection_model.h5' in {SCRIPT_DIR}"
        )

    print(f"[DEEPSHIELD-PY] Loading video model: {H5_MODEL_PATH}")
    _video_model = build_video_model()
    print("[DEEPSHIELD-PY] Video model loaded!")
    return _video_model


# ============================================================
# Image Prediction  (HuggingFace)
# ============================================================

def predict_image(image_path):
    """Classify a single image using the HuggingFace ViT model."""
    classifier = get_image_classifier()

    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        results = classifier(img)
    except Exception as e:
        return {
            'prediction': 'ERROR',
            'confidence': 0,
            'message': f'Detection failed: {str(e)}',
            'error': str(e)
        }

    # Parse HuggingFace results
    # Labels: 'artificial' and 'human'
    scores = {}
    for item in results:
        label = item['label'].lower()
        score = round(item['score'] * 100, 2)
        if label in ('artificial', 'ai_generated', 'fake'):
            scores['FAKE'] = score
        elif label in ('human', 'real'):
            scores['REAL'] = score
        else:
            scores[label] = score

    real_prob = scores.get('REAL', 0)
    fake_prob = scores.get('FAKE', 0)

    # Determine prediction
    if fake_prob > real_prob:
        if fake_prob >= 75:
            prediction = 'AI-GENERATED'
            confidence = fake_prob
            gen_prob = fake_prob
            edited_prob = 0
        else:
            prediction = 'AI-EDITED'
            confidence = fake_prob
            gen_prob = 0
            edited_prob = fake_prob
    else:
        prediction = 'REAL'
        confidence = real_prob
        gen_prob = 0
        edited_prob = 0

    message = MESSAGES.get(prediction, f'Classification: {prediction}')

    return {
        'prediction': prediction,
        'confidence': confidence,
        'message': message,
        'real_probability': real_prob,
        'edited_probability': edited_prob,
        'generated_probability': gen_prob,
        'models_used': {
            'huggingface_vit': True,
            'xception_lstm': False,
            'ela': False,
            'segmenter': False,
        },
        'model_details': {
            'model': HF_MODEL_NAME,
            'type': 'image',
            'raw_scores': {item['label']: round(item['score'] * 100, 2) for item in results}
        }
    }


# ============================================================
# Video Prediction  (Local H5 — Xception + LSTM)
# ============================================================

def predict_video(video_path):
    """Extract 10 frames from the video and classify as one sequence."""
    import cv2
    import numpy as np

    model = get_video_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'prediction': 'ERROR',
            'confidence': 0,
            'message': 'Could not open video file.',
            'error': 'Failed to open video'
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps if fps > 0 else 0

    # Select 10 evenly-spaced frame indices
    if total_frames <= SEQUENCE_LENGTH:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / SEQUENCE_LENGTH
        frame_indices = [int(step * i) for i in range(SEQUENCE_LENGTH)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frames.append(frame_norm)

    cap.release()

    if not frames:
        return {
            'prediction': 'ERROR',
            'confidence': 0,
            'message': 'Could not extract any frames from video.',
            'error': 'No frames extracted'
        }

    # Pad or trim to exactly SEQUENCE_LENGTH frames
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])
    frames = frames[:SEQUENCE_LENGTH]

    sequence = np.stack(frames, axis=0)
    batch = np.expand_dims(sequence, axis=0)  # (1, 10, 128, 128, 3)

    try:
        preds = model.predict(batch, verbose=0)
    except Exception as e:
        return {
            'prediction': 'ERROR',
            'confidence': 0,
            'message': f'Detection failed: {str(e)}',
            'error': str(e)
        }

    real_prob = round(float(preds[0][0]) * 100, 2)
    fake_prob = round(float(preds[0][1]) * 100, 2)

    # Determine prediction
    if fake_prob > real_prob:
        if fake_prob >= 75:
            prediction = 'AI-GENERATED'
            gen_prob = fake_prob
            edited_prob = 0
        else:
            prediction = 'AI-EDITED'
            gen_prob = 0
            edited_prob = fake_prob
        confidence = fake_prob
    else:
        prediction = 'REAL'
        confidence = real_prob
        gen_prob = 0
        edited_prob = 0

    if prediction == 'AI-GENERATED':
        message = f'This video appears to be AI-generated or a deepfake. Confidence: {confidence}%.'
    elif prediction == 'AI-EDITED':
        message = f'This video shows signs of AI editing/manipulation. Confidence: {confidence}%.'
    else:
        message = f'This video appears to be authentic. Confidence: {confidence}%.'

    return {
        'prediction': prediction,
        'confidence': confidence,
        'message': message,
        'real_probability': real_prob,
        'edited_probability': edited_prob,
        'generated_probability': gen_prob,
        'video_info': {
            'total_frames': total_frames,
            'fps': round(fps, 2),
            'duration_seconds': round(duration, 2),
            'frames_analyzed': len(frame_indices)
        },
        'models_used': {
            'huggingface_vit': False,
            'xception_lstm': True,
            'ela': False,
            'segmenter': False,
        },
        'model_details': {
            'model': 'deepfake_detection_model.h5',
            'type': 'video',
            'architecture': 'Xception + LSTM',
            'raw_scores': {'real': real_prob, 'fake': fake_prob}
        }
    }


# ============================================================
# Flask Server
# ============================================================

def run_server():
    """Persistent Flask server — models load once, stay in memory."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # Models are lazy-loaded on first request for fast server startup
    print("[DEEPSHIELD-PY] Server ready on port 5001 (models will load on first use).")

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok',
            'image_model_loaded': _image_classifier is not None,
            'video_model_loaded': _video_model is not None
        })

    @app.route('/detect', methods=['POST'])
    def detect():
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing file_path'}), 400
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        result = predict_image(file_path)
        return jsonify(result)

    @app.route('/detect-video', methods=['POST'])
    def detect_video():
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing file_path'}), 400
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        result = predict_video(file_path)
        return jsonify(result)

    app.run(host='127.0.0.1', port=5001, debug=False)


# ============================================================
# Main
# ============================================================

def main():
    if '--server' in sys.argv:
        run_server()
        return

    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Usage: python detect.py <file_path> [--video] OR python detect.py --server'
        }))
        sys.exit(1)

    file_path = sys.argv[1]
    is_video = '--video' in sys.argv

    if not os.path.exists(file_path):
        print(json.dumps({'error': f'File not found: {file_path}'}))
        sys.exit(1)

    try:
        if is_video:
            result = predict_video(file_path)
        else:
            result = predict_image(file_path)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({
            'prediction': 'ERROR',
            'confidence': 0,
            'message': f'Detection failed: {str(e)}',
            'error': str(e)
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
