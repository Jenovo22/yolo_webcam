import time
import threading
from pathlib import Path

import cv2
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO


# ==========================
# CONFIGURACIÓN GENERAL
# ==========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

CAMERA_INDEX = 0

HOST = "0.0.0.0"
PORT = 3001

CONFIDENCE = 0.50
IMG_SIZE = 416

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 75

SHOW_FPS = True


# ==========================
# VALIDACIONES
# ==========================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")


# ==========================
# APP FLASK
# ==========================

app = Flask(__name__)


# ==========================
# CARGA DEL MODELO
# ==========================

print("Cargando modelo YOLO...")
model = YOLO(str(MODEL_PATH))
print("Modelo cargado correctamente.")


# ==========================
# CÁMARA GLOBAL
# ==========================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    raise RuntimeError(
        "No se pudo abrir la cámara. "
        "Prueba cambiando CAMERA_INDEX a 1 o revisa que la cámara esté conectada."
    )


camera_lock = threading.Lock()


# ==========================
# ESTADO DEL SISTEMA
# ==========================

last_fps = 0.0
last_inference_ms = 0.0
last_error = None


# ==========================
# GENERADOR DE FRAMES
# ==========================

def gen_frames():
    global last_fps, last_inference_ms, last_error

    prev_time = time.time()

    while True:
        try:
            with camera_lock:
                success, frame = camera.read()

            if not success or frame is None:
                last_error = "No se pudo leer frame de la cámara."
                time.sleep(0.1)
                continue

            inference_start = time.time()

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                verbose=False
            )

            annotated_frame = results[0].plot()

            inference_end = time.time()
            last_inference_ms = (inference_end - inference_start) * 1000

            current_time = time.time()
            elapsed = current_time - prev_time

            if elapsed > 0:
                last_fps = 1.0 / elapsed

            prev_time = current_time

            if SHOW_FPS:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {last_fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"Inferencia: {last_inference_ms:.0f} ms",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            ret, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            if not ret:
                last_error = "No se pudo codificar el frame como JPEG."
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        except GeneratorExit:
            break

        except Exception as e:
            last_error = str(e)
            print(f"Error en gen_frames: {e}")
            time.sleep(0.2)


# ==========================
# RUTAS WEB
# ==========================

@app.route("/")
def index():
    return render_template(
        "index.html",
        confidence=CONFIDENCE,
        img_size=IMG_SIZE,
        port=PORT
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    return jsonify({
        "model": str(MODEL_PATH),
        "camera_index": CAMERA_INDEX,
        "confidence": CONFIDENCE,
        "img_size": IMG_SIZE,
        "fps": round(last_fps, 2),
        "inference_ms": round(last_inference_ms, 2),
        "last_error": last_error
    })


# ==========================
# EJECUCIÓN
# ==========================

if __name__ == "__main__":
    print(f"Servidor iniciado en http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)