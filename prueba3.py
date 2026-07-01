import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Diccionario con la configuración de las zonas y rutas de video
configuraciones = {
    "5": {
        "video_path": "muestras/video5.mp4",
        "zona_a": [[1264, 323], [1906, 173], [1900, 1055], [8, 1058], [9, 223], [628, 333]],
        "zona_b": [[1262, 293], [1906, 50], [1905, 0], [4, 5], [8, 102], [648, 303]]
    },
    "6": {
        "video_path": "muestras/video6.mp4",
        "zona_a": [[795, 5], [801, 789], [447, 1062], [10, 1055], [4, 3]],
        "zona_b": [[812, 3], [846, 792], [576, 1047], [1904, 1055], [1904, 10]]
    }
}

print("=" * 50)
print("  SELECCION DE VIDEO Y ZONAS")
print("=" * 50)
opcion = input("Seleccione el video a procesar (5 o 6): ").strip()

if opcion not in configuraciones:
    print("Opción no válida. Saliendo...")
    exit()

config = configuraciones[opcion]
video_path = config["video_path"]
polygon_a = np.array(config["zona_a"], np.int32)
polygon_b = np.array(config["zona_b"], np.int32)

print(f"Procesando video {opcion}")
print(f"Zona A definida con {len(polygon_a)} puntos")
print(f"Zona B definida con {len(polygon_b)} puntos")

# MODELO
model = YOLO("models/model_v8.2.onnx")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video {video_path}.")
    exit()

# CREAR ZONAS (triggering_anchors=CENTER para que solo cuente cuando el centro de la caja entre en la zona)
zone_a = sv.PolygonZone(polygon=polygon_a, triggering_anchors=[sv.Position.CENTER])
zone_b = sv.PolygonZone(polygon=polygon_b, triggering_anchors=[sv.Position.CENTER])

zone_a_annotator = sv.PolygonZoneAnnotator(zone=zone_a, thickness=2, text_thickness=1, text_scale=0.5, color=sv.Color.RED)
zone_b_annotator = sv.PolygonZoneAnnotator(zone=zone_b, thickness=2, text_thickness=1, text_scale=0.5, color=sv.Color.BLUE)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

# CONTADORES
track_states = {}
entradas = 0
salidas = 0

print("\nProcesando video... Presiona 'q' para salir.\n")

# Crear ventana redimensionable que se ajusta a la pantalla
cv2.namedWindow("Conteo en Autobus", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Conteo en Autobus", 1280, 720)

frame_count = 0
frame_skip = 2  # Procesar 1 de cada 2 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # RESULTADOS DE LA DETECCION
    results = model.track(frame, persist=True, tracker="custom_botsort.yaml", imgsz=640, verbose=False, conf=0.35)[0]

    # CONVERTIR A FORMATO DE SUPERVISON
    detections = sv.Detections.from_ultralytics(results)

    # SOLO CABEZAS
    detections = detections[np.isin(detections.class_id, [0])]

    if detections.tracker_id is not None:

        # VERIFICAR QUE ESTEN DENTRO DE CADA ZONA
        mask_a = zone_a.trigger(detections=detections)
        mask_b = zone_b.trigger(detections=detections)

        # LOGICA DE CONTEO
        for i, track_id in enumerate(detections.tracker_id):
            in_zone_a = mask_a[i]
            in_zone_b = mask_b[i]

            # ESTADO ACTUAL DE ESTE ID EN NUESTRA MEMORIA
            estado_previo = track_states.get(track_id)

            if in_zone_a and not in_zone_b:
                if estado_previo == "ZONA_B":
                    # TRANSCICION B->A: ENTRADA CONFIRMADA
                    entradas += 1
                track_states[track_id] = "ZONA_A"

            elif in_zone_b and not in_zone_a:
                if estado_previo == "ZONA_A":
                    # TRANSCICION A->B: SALIDA CONFIRMADA
                    salidas += 1
                track_states[track_id] = "ZONA_B"

    # Visualización
    frame = zone_a_annotator.annotate(scene=frame)
    frame = zone_b_annotator.annotate(scene=frame)

    # DIBUJAR LAS CAJAS Y LOS IDs
    frame = box_annotator.annotate(scene=frame, detections=detections)
    if detections.tracker_id is not None:
        labels = [
            f"ID: {tracker_id} Conf: {conf:.2f}"
            for tracker_id, conf in zip(detections.tracker_id, detections.confidence)
        ]
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Mostrar contadores y métricas en pantalla
    cv2.putText(frame, f"Entradas: {entradas}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Salidas: {salidas}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Métricas de rendimiento
    infer_time = results.speed.get('inference', 0.0)
    total_time = sum(results.speed.values())
    fps = 1000.0 / total_time if total_time > 0 else 0.0
    
    cv2.putText(frame, f"Inferencia: {infer_time:.1f} ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS Modelo: {fps:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Conteo en Autobus", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Mostrar resultado final
print(f"\n{'=' * 50}")
print(f"  RESULTADO FINAL")
print(f"{'=' * 50}")
print(f"  Entradas: {entradas}")
print(f"  Salidas:  {salidas}")
print(f"{'=' * 50}")
