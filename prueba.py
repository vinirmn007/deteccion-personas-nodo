import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import sys

# Función para seleccionar una zona (polígono) con el mouse
def seleccionar_zona(frame, nombre_zona, color):
    """
    Muestra el frame y permite al usuario dibujar un polígono
    haciendo clic en los vértices.
    """
    puntos = []
    ventana = f"Dibuja {nombre_zona} - Enter para confirmar"

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()

    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ventana, mouse_callback)

    while True:
        display = frame.copy()

        # Dibujar instrucciones en pantalla
        cv2.putText(display, f"Dibujando: {nombre_zona}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, "Clic izq: agregar punto | Clic der: deshacer", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Enter/Espacio: confirmar | Esc: cancelar", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Puntos: {len(puntos)} (minimo 3)", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Dibujar los puntos y las líneas del polígono
        for i, pt in enumerate(puntos):
            cv2.circle(display, pt, 5, color, -1)
            cv2.putText(display, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if i > 0:
                cv2.line(display, puntos[i - 1], pt, color, 2)

        # Cerrar el polígono visualmente si hay 3+ puntos
        if len(puntos) >= 3:
            cv2.line(display, puntos[-1], puntos[0], color, 2)
            overlay = display.copy()
            pts_array = np.array(puntos, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_array], (*color, ))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        cv2.imshow(ventana, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):  # Enter o Espacio
            if len(puntos) >= 3:
                break
        elif key == 27:  # Esc
            cv2.destroyWindow(ventana)
            return None

    cv2.destroyWindow(ventana)
    return np.array(puntos)


# MODELO
model = YOLO("models/model_v8.2.onnx")

# Video
cap = cv2.VideoCapture("muestras/video5.mp4")

ret, primer_frame = cap.read()
if not ret:
    print("Error: No se pudo leer el video.")
    cap.release()
    sys.exit()

print("=" * 50)
print("  SELECCION DE ZONAS")
print("=" * 50)
print("  1) Dibuja la ZONA A (Adentro del bus)")
print("  2) Luego la ZONA B (Puerta/Escalones)")
print("  Clic izquierdo = agregar punto")
print("  Clic derecho   = deshacer punto")
print("  Enter/Espacio  = confirmar zona")
print("=" * 50)

# ZONA A (ADENTRO) 
polygon_a = seleccionar_zona(primer_frame, "ZONA A (Adentro)", (0, 0, 255))
if polygon_a is None:
    print("Seleccion cancelada.")
    cap.release()
    sys.exit()
print(f"Zona A definida con {len(polygon_a)} puntos: {polygon_a.tolist()}")

# ZONA B (PUERTA)
polygon_b = seleccionar_zona(primer_frame, "ZONA B (Puerta)", (255, 0, 0))
if polygon_b is None:
    print("Seleccion cancelada.")
    cap.release()
    sys.exit()
print(f"Zona B definida con {len(polygon_b)} puntos: {polygon_b.tolist()}")

# CREAR ZONAS
zone_a = sv.PolygonZone(polygon=polygon_a, triggering_anchors=[sv.Position.CENTER])
zone_b = sv.PolygonZone(polygon=polygon_b, triggering_anchors=[sv.Position.CENTER])

# CONTADORES
track_states = {}
entradas = 0
salidas = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nProcesando video de {total_frames} frames en segundo plano... Por favor espera.\n")

frame_count = 0
frame_skip = 2  # Procesar 1 de cada 2 frames para optimizar

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Imprimir progreso en la misma línea
    if frame_count % 10 == 0:
        print(f"Procesando frame {frame_count} / {total_frames} ...", end="\r")

    if frame_count % frame_skip != 0:
        continue

    # RESULTADOS DE LA DETECCION
    results = model.track(frame, persist=True, tracker="custom_botsort.yaml", imgsz=640, verbose=False, conf=0.35)[0]

    # CONVERTIR A FORMATO DE SUPERVISON
    detections = sv.Detections.from_ultralytics(results)

    # SOLO CABEZAS (clase 0 en el modelo)
    detections = detections[np.isin(detections.class_id, [0])]

    if detections.tracker_id is not None:
        # VERIFICAR QUE ESTEN DENTRO DE CADA ZONA
        mask_a = zone_a.trigger(detections=detections)
        mask_b = zone_b.trigger(detections=detections)

        # LOGICA DE CONTEO
        for i, track_id in enumerate(detections.tracker_id):
            in_zone_a = mask_a[i]
            in_zone_b = mask_b[i]

            estado_previo = track_states.get(track_id)

            if in_zone_a and not in_zone_b:
                if estado_previo == "ZONA_B":
                    entradas += 1
                track_states[track_id] = "ZONA_A"

            elif in_zone_b and not in_zone_a:
                if estado_previo == "ZONA_A":
                    salidas += 1
                track_states[track_id] = "ZONA_B"

cap.release()
print("\n")  # Salto de línea para limpiar el \r del progreso

# Mostrar resultado final
print(f"{'=' * 50}")
print(f"  RESULTADO FINAL")
print(f"{'=' * 50}")
print(f"  Entradas: {entradas}")
print(f"  Salidas:  {salidas}")
print(f"{'=' * 50}")