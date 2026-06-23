from ultralytics import YOLO

def export_to_onnx():
    print("Cargando el modelo PyTorch...")
    # Asegúrate de que la ruta al modelo sea la correcta
    model = YOLO("models/model_v8.pt")
    model.eval()  # Establecer el modelo en modo evaluación

    print("Iniciando exportación a ONNX...")
    # Exportar a ONNX. 
    # half=True (opcional) exporta el modelo en precisión FP16, haciéndolo aún más ligero y rápido.
    # dynamic=True (opcional) permite tamaños de imagen de entrada dinámicos.
    path = model.export(format="onnx", imgsz=640, half=True, dynamic=True)
    
    print(f"Exportación completada. El modelo ONNX se encuentra en: {path}")

if __name__ == "__main__":
    export_to_onnx()
