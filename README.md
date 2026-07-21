# Detección y Conteo de Personas en Autobuses (YOLOv8 + Supervision + Tracking)

Este proyecto implementa un sistema automatizado de conteo de pasajeros (entradas y salidas) en autobuses utilizando modelos de visión por computador basado en **YOLOv8**, la librería **Supervision** de Roboflow y rastreo multiobjeto (**BoT-SORT** / **ByteTrack**).

Está diseñado y optimizado para ejecutarse tanto en entornos de desarrollo (PC/Laptop) como en dispositivos embebidos de alto rendimiento (**NVIDIA Jetson Nano**).

---

## 🚀 Flujo de Ejecución y Configuración Inicial

Sigue esta secuencia paso a paso para clonar el repositorio, preparar el entorno y ejecutar las pruebas:

### 1. Clonar el Repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd prueba1_yolov26
```

### 2. Instalar Dependencias
* **En PC / Laptop / Servidor:**
  Instala los paquetes necesarios utilizando el archivo de requerimientos estándar:
  ```bash
  pip install -r requirements.txt
  ```

* **En NVIDIA Jetson (Jetson Nano / Xavier / Orin):**
  > 📌 **Nota para Jetson:** Existe un archivo de dependencias compilado específicamente para la arquitectura y entorno de la Jetson (con soporte PyTorch/Torchvision acelerados por CUDA para JetPack). Este archivo se encuentra almacenado directamente en la Jetson para su importación e instalación local.

### 3. Descargar Videos de Muestra
El proyecto incluye un script automatizado para descargar los videos de prueba almacenados en Google Drive hacia la carpeta local `muestras/`:

```bash
python download_drive.py
```
*Este script verificar e instalará automáticamente `gdown` si no lo tienes instalado y creará el directorio `muestras/` con los archivos de video necesarios.*

---

## 🧪 Modos de Prueba y Uso

### 🎨 Prueba 2 (`prueba2.py`): Selección Manual Interactiva de Zonas

En esta prueba puedes **dibujar las zonas de conteo de forma interactiva con el mouse** sobre el primer fotograma del video antes de iniciar el procesamiento.

* **Controles para dibujar zonas:**
  * **Clic Izquierdo:** Agregar punto al polígono.
  * **Clic Derecho:** Deshacer último punto.
  * **Enter / Espacio:** Confirmar zona (mínimo 3 puntos).
  * **Esc:** Cancelar.
  *(Primero se dibuja la ZONA A - Adentro del bus, y luego la ZONA B - Puerta/Escalones).*

* **¿Cómo cambiar el video en `prueba2.py`?**
  Abre el archivo [prueba2.py](file:///d:/prueba1_yolov26/prueba2.py) y modifica la línea 76:
  ```python
  # Línea 76 de prueba2.py:
  cap = cv2.VideoCapture("muestras/video5.mp4")  # Cambia la ruta o coloca 0 para usar webcam
  ```

---

### ⚙️ Prueba 3 (`prueba3.py`): Zonas Pre-anotadas y Selección por Argumentos

En esta prueba las zonas ya están **pre-anotadas y calibradas** en el código (dentro del diccionario `configuraciones`). Puedes elegir qué video procesar directamente desde la terminal.

* **Sintaxis de ejecución:**
  ```bash
  python prueba3.py --video <NUMERO_VIDEO>
  ```

* **Opciones disponibles:**
  * `python prueba3.py --video 1` : Procesa `muestras/video1.mp4` (o `.webm`)
  * `python prueba3.py --video 2` : Procesa `muestras/video2.mp4`
  * `python prueba3.py --video 3` : Procesa `muestras/video3.mp4` (o `.dav`)
  * `python prueba3.py --video 4` : Procesa `muestras/video4.mp4`

* **Modo Headless (Ideal para Jetson Nano / Sistemas sin pantalla):**
  Puedes agregar la bandera `--headless` para omitir la renderización gráfica de OpenCV y acelerar el procesamiento:
  ```bash
  python prueba3.py --video 1 --headless
  ```

---

## ⚡ Formatos de Modelo: PyTorch (`.pt`), ONNX (`.onnx`) y TensorRT (`.engine`)

### ¿Qué es ONNX y TensorRT Engine?

* **ONNX (`.onnx` - Open Neural Network Exchange):** Formato abierto de representación de modelos de Deep Learning. Permite portar modelos entrenados en PyTorch a múltiples entornos de inferencia multiplataforma de forma eficiente.
* **TensorRT Engine (`.engine`):** Formato binario compilado y ultranivel optimizado por NVIDIA específicamente para su arquitectura de hardware GPU.

### Convertir un Modelo Recién Entrenado (`.pt`)

Si entrenas o actualizas un modelo PyTorch (`.pt`) y necesitas convertirlo para inferencia rápida:

1. **Exportar a ONNX:**
   ```bash
   python export_onnx.py
   ```
2. **Exportar a TensorRT / ONNX optimizado:**
   ```bash
   python export_tensorrt.py          # Para conversión directa a TensorRT (requiere GPU NVIDIA)
   python export_tensorrt.py --onnx   # Exporta ONNX optimizado para llevar a la Jetson Nano
   ```

### 🏎️ ¿Por qué se utiliza `.engine` en NVIDIA Jetson?

El formato `.engine` (TensorRT) es esencial para desplegar modelos en la **NVIDIA Jetson Nano**:
1. **Máxima Optimización de Hardware:** TensorRT realiza fusión de capas (layer fusion), selección de kernels optimizados y cuantización a precisión **FP16** (Half Precision).
2. **Baja Latencia y Alto Rendimiento:** Reduce dramáticamente el consumo de memoria RAM/VRAM y multiplica los FPS de inferencia en hardware embebido con recursos limitados.
3. **Importante sobre la Portabilidad:** Los archivos `.engine` **NO son portables entre diferentes modelos de GPU**. Un archivo `.engine` generado en una PC con GPU RTX no funcionará en una Jetson Nano. Por ello, el procedimiento recomendado es exportar a `.onnx` y generar el `.engine` directamente **DENTRO de la Jetson Nano**.

---

## 🎯 Configuración del Tracker (BoT-SORT / ByteTrack)

El sistema utiliza algoritmos de rastreo multiobjeto para mantener la identidad de las personas a lo largo de las secuencias de video.

* **BoT-SORT Personalizado (`custom_botsort.yaml`):**
  Por defecto se utiliza la configuración [custom_botsort.yaml](file:///d:/prueba1_yolov26/custom_botsort.yaml), la cual ha sido ajustada para cámaras fijas (desactivando la compensación de movimiento global `gmc_method: none` y ReID `with_reid: False`) logrando menor carga computacional y mayor estabilidad en el tracking de pasajeros.

* **Cambio a ByteTrack (`bytetrack.yaml`):**
  Si se requiere, se puede cambiar el rastreador a ByteTrack modificando la llamada de rastreo en el modelo a `tracker="bytetrack.yaml"`.

---

## ⚠️ Scripts Obsoletos (`train.py` y `deploy.py`)

> ℹ️ **Nota:** Los archivos `train.py` y `deploy.py` corresponden a versiones previas de pruebas de entrenamiento y despliegue. No requieren configuración ni uso, ya que están marcados para ser descontinuados o eliminados del proyecto.
