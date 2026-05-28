# Implementación de cámara en tiempo real con YOLO en Raspberry Pi

Este procedimiento permite ejecutar en una Raspberry Pi una aplicación web que muestra en tiempo real la cámara conectada a la Raspberry y procesa cada frame usando un modelo de inteligencia artificial `.pt` con Ultralytics YOLO.

---

## 1. Conectarse a la Raspberry Pi por SSH

Desde PowerShell en Windows:

```powershell
ssh usuario@test.local
```

Ejemplo:

```powershell
ssh jeronimorp@test.local
```

Si aparece un error como `REMOTE HOST IDENTIFICATION HAS CHANGED`, limpiar la clave anterior:

```powershell
ssh-keygen -R test.local
```

Luego volver a conectarse:

```powershell
ssh jeronimorp@test.local
```

---

## 2. Crear la carpeta del proyecto en la Raspberry Pi

Dentro de la Raspberry:

```bash
mkdir -p ~/yolo_webcam/templates
cd ~/yolo_webcam
```

La estructura final del proyecto será:

```text
yolo_webcam/
├── app.py
├── best.pt
└── templates/
    └── index.html
```

---

## 3. Subir el modelo `.pt` desde el computador local a la Raspberry

Desde PowerShell en Windows, ejecutar:

```powershell
scp "C:\Users\jeron\Downloads\best.pt" jeronimorp@test.local:~/yolo_webcam/best.pt
```

Si el modelo está en otra carpeta, cambiar la ruta local.

Verificar que el modelo quedó en la Raspberry:

```bash
ls -lh ~/yolo_webcam/best.pt
```

---

## 4. Instalar dependencias del sistema

En la Raspberry:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
```

---

## 5. Crear y activar entorno virtual

```bash
cd ~/yolo_webcam
python3 -m venv .venv
source .venv/bin/activate
```

Actualizar `pip`:

```bash
pip install --upgrade pip
```

---

## 6. Instalar librerías de Python

```bash
pip install flask ultralytics opencv-python-headless
```

Librerías principales utilizadas:

```text
flask
ultralytics
opencv-python-headless
torch
numpy
```

`torch` y `numpy` normalmente se instalan como dependencias de `ultralytics`.

---

## 7. Crear el archivo principal `app.py`

Crear el archivo:

```bash
nano ~/yolo_webcam/app.py
```

Pegar el siguiente código:

```python
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
```

Guardar con:

```text
Ctrl + O
Enter
Ctrl + X
```

---

## 8. Crear la página web `index.html`

Crear el archivo:

```bash
nano ~/yolo_webcam/templates/index.html
```

Pegar el siguiente código:

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Raspberry Pi - YOLO en tiempo real</title>

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #111;
            color: #f5f5f5;
            text-align: center;
        }

        header {
            padding: 20px;
            background: #1f1f1f;
            border-bottom: 1px solid #333;
        }

        h1 {
            margin: 0;
            font-size: 28px;
        }

        .subtitle {
            color: #aaa;
            margin-top: 8px;
        }

        .container {
            padding: 24px;
        }

        .video-box {
            display: inline-block;
            background: #222;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #444;
        }

        img {
            width: 90vw;
            max-width: 960px;
            border-radius: 8px;
        }

        .info {
            margin-top: 18px;
            color: #ccc;
            font-size: 15px;
        }

        .status {
            margin-top: 12px;
            color: #8ee88e;
            font-size: 14px;
        }

        code {
            background: #222;
            padding: 3px 6px;
            border-radius: 4px;
            color: #ddd;
        }
    </style>
</head>

<body>
    <header>
        <h1>Detección en tiempo real con YOLO</h1>
        <div class="subtitle">Raspberry Pi + Cámara + Modelo .pt</div>
    </header>

    <div class="container">
        <div class="video-box">
            <img src="{{ url_for('video_feed') }}" alt="Video en tiempo real">
        </div>

        <div class="info">
            <p>Modelo cargado desde: <code>best.pt</code></p>
            <p>Confianza: <code>{{ confidence }}</code> | Tamaño de inferencia: <code>{{ img_size }}</code></p>
            <p>Servidor: <code>http://IP_DE_LA_RASPBERRY:{{ port }}</code></p>
        </div>

        <div class="status" id="status">
            Cargando estado...
        </div>
    </div>

    <script>
        async function updateStatus() {
            try {
                const response = await fetch("/status");
                const data = await response.json();

                const status = document.getElementById("status");

                status.innerHTML = `
                    FPS: ${data.fps} |
                    Inferencia: ${data.inference_ms} ms |
                    Cámara: ${data.camera_index}
                    ${data.last_error ? "<br>Error: " + data.last_error : ""}
                `;
            } catch (error) {
                document.getElementById("status").innerText =
                    "No se pudo obtener el estado del sistema.";
            }
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
```

Guardar con:

```text
Ctrl + O
Enter
Ctrl + X
```

---

## 9. Verificar estructura del proyecto

```bash
cd ~/yolo_webcam
ls -lah
ls -lah templates
```

Debe existir:

```text
~/yolo_webcam/app.py
~/yolo_webcam/best.pt
~/yolo_webcam/templates/index.html
```

---

## 10. Verificar cámara

Listar dispositivos de video:

```bash
ls /dev/video*
```

Probar lectura de cámara:

```bash
cd ~/yolo_webcam
source .venv/bin/activate

python - <<'PY'
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print("Camara abierta:", cap.isOpened())

ret, frame = cap.read()
print("Frame leido:", ret)

cap.release()
PY
```

Si `Camara abierta` aparece como `False`, editar `app.py` y cambiar:

```python
CAMERA_INDEX = 0
```

por:

```python
CAMERA_INDEX = 1
```

Luego volver a probar.

---

## 11. Ejecutar la aplicación

```bash
cd ~/yolo_webcam
source .venv/bin/activate
python app.py
```

Si todo está correcto, aparecerá algo similar a:

```text
Servidor iniciado en http://0.0.0.0:3001
Running on http://127.0.0.1:3001
Running on http://192.168.0.113:3001
```

---

## 12. Abrir la página desde otro computador

Desde el navegador del computador conectado a la misma red:

```text
http://test.local:3001
```

O usando la IP de la Raspberry:

```text
http://192.168.0.113:3001
```

Cambiar la IP por la que muestre la Raspberry al iniciar Flask.

---

## 13. Solución de error común: `TemplateNotFound: index.html`

Si aparece:

```text
jinja2.exceptions.TemplateNotFound: index.html
```

significa que Flask no encontró el archivo HTML.

Verificar:

```bash
ls -lah ~/yolo_webcam/templates/index.html
```

Si `index.html` quedó directamente en `~/yolo_webcam`, moverlo:

```bash
cd ~/yolo_webcam
mkdir -p templates
mv index.html templates/index.html
```

Luego reiniciar:

```bash
python app.py
```

---

## 14. Ajustes de rendimiento recomendados

Si la Raspberry va lenta, editar `app.py` y bajar estos valores:

```python
IMG_SIZE = 320
FRAME_WIDTH = 480
FRAME_HEIGHT = 360
```

Configuración inicial recomendada:

```python
IMG_SIZE = 416
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE = 0.50
```

---

## 15. Detener la aplicación

En la terminal donde está corriendo Flask:

```text
Ctrl + C
```

---

## 16. Ejecutar nuevamente después de reiniciar la Raspberry

```bash
cd ~/yolo_webcam
source .venv/bin/activate
python app.py
```

Abrir en el navegador:

```text
http://test.local:3001
```

o:

```text
http://IP_DE_LA_RASPBERRY:3001
```#   y o l o _ w e b c a m  
 