import os
import argparse
from ultralytics import YOLO
from roboflow import Roboflow

def train_yolo():
    parser = argparse.ArgumentParser(description="Entrenamiento local de YOLO optimizado para baja luz.")
    parser.add_argument('--api_key', type=str, default="k1lfw5vNH559hQizTZot", help="API Key de Roboflow")
    parser.add_argument('--model', type=str, default="yolov8s.pt", help="Modelo base de YOLO (ej. yolov8n.pt, yolov8s.pt)")
    parser.add_argument('--epochs', type=int, default=100, help="Número de épocas de entrenamiento")
    parser.add_argument('--batch', type=int, default=16, help="Tamaño de lote (batch size)")
    parser.add_argument('--imgsz', type=int, default=640, help="Tamaño de las imágenes de entrada")
    parser.add_argument('--device', type=str, default="0", help="Dispositivo para entrenar (ej. 0, cpu, mps)")
    parser.add_argument('--force_download', action='store_true', help="Forzar la descarga del dataset desde Roboflow")
    args = parser.parse_args()

    # Limpiar variables de entorno que puedan interferir en ejecuciones distribuidas
    os.environ.pop('LOCAL_RANK', None)
    os.environ.pop('RANK', None)
    os.environ.pop('WORLD_SIZE', None)

    # 1. Obtención del dataset desde Roboflow (similar a train.py)
    dataset_dir = "proyectonodo-7"
    dataset_yaml = os.path.join(os.getcwd(), dataset_dir, "data.yaml")

    if not os.path.exists(dataset_yaml) or args.force_download:
        print("Descargando dataset desde Roboflow...")
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace("alexiss-workspace-sentr").project("proyectonodo")
        version = project.version(7)
        dataset = version.download("yolo26")
        dataset_yaml = os.path.join(dataset.location, "data.yaml")
    else:
        print(f"El dataset ya existe localmente en: {dataset_dir}. Omitiendo descarga (usa --force_download para forzar).")

    # 2. Configuración y Carga del Modelo Base
    # Usamos por defecto 'yolov8s.pt' para una mejor capacidad de extracción de características en baja luz.
    print(f"Cargando modelo base: {args.model} ...")
    model = YOLO(args.model)

    # 3. Entrenamiento con Aumentos de Datos para Baja Luz
    print("Iniciando entrenamiento local con simulaciones de baja luz...")
    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project='runs/detect',
        name='modelo_bajo_luz',
        # --- AUMENTOS DE DATOS PARA BAJA LUZ Y CAPUCHAS ---
        hsv_v=0.6,      # Variación de brillo (default: 0.4). Ayuda a detectar en penumbra extrema.
        hsv_s=0.4,      # Reduce la saturación (default: 0.7) simulando pérdida de color nocturna.
        blur=0.3,       # Introduce desenfoque (default: 0.0) simulando obturación lenta y ruido.
        degrees=10.0,   # Pequeñas rotaciones para robustez en inclinaciones de cabeza.
        scale=0.5,      # Variación de escala para detectar cabezas a distintas distancias.
        mosaic=1.0,     # Mosaico activo para robustez frente a oclusiones y multitudes.
        # --------------------------------------------------
    )

if __name__ == '__main__':
    train_yolo()
