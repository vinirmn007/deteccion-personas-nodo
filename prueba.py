from ultralytics import YOLO

# 1. Cargar TU modelo entrenado
model = YOLO('model_v3.pt')

# 2. Ejecutar la cámara web en tiempo real

resultados = model.predict(
    source="muestras/video1.webm", 
    show=True,     # Muestra la ventana de video
    conf=0.5,      # Umbral de confianza: Solo muestra cajas con más de 50% de seguridad
    stream=True    # Recomendado para videos largos o cámaras en vivo
)

# 3. Mantener el programa corriendo
for r in resultados:
    pass