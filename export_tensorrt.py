from ultralytics import YOLO

def export_to_tensorrt():
    """
    Exporta el modelo YOLO a formato TensorRT (.engine) para NVIDIA Jetson Nano.
    
    IMPORTANTE para Jetson Nano:
    - Usar half=True (FP16) ya que la Jetson Nano soporta FP16 con buen rendimiento.
    - Usar imgsz=640 o menor (320 es más rápido en Jetson Nano).
    - El archivo .engine generado en una PC NO es portátil a Jetson Nano directamente.
      Debes generar el .engine EN la Jetson Nano, o exportar primero a ONNX y 
      luego convertir a TensorRT en la Jetson.
    """
    print("Cargando el modelo PyTorch...")
    model = YOLO("models/model_v8.2.pt")

    # --- OPCION A: Exportar directamente a TensorRT (solo funciona si tienes GPU NVIDIA) ---
    print("Exportando a TensorRT (.engine)...")
    path = model.export(
        format="engine",   # Formato TensorRT
        imgsz=640,          # Tamaño de imagen (usa 320 para más velocidad en Jetson)
        half=True,          # FP16 - ideal para Jetson Nano
        simplify=True,      # Simplificar el grafo ONNX intermedio
        workspace=2.0,      # GB de workspace para TensorRT (Jetson Nano tiene 4GB RAM)
    )
    print(f"Exportación completada: {path}")


def export_to_onnx_for_jetson():
    """
    Exporta a ONNX optimizado para luego convertir a TensorRT EN la Jetson Nano.
    
    Este es el método MÁS RECOMENDADO porque:
    - Los archivos .engine NO son portátiles entre diferentes GPUs.
    - Puedes generar el ONNX en cualquier PC y luego convertirlo en la Jetson.
    """
    print("Cargando el modelo PyTorch...")
    model = YOLO("models/model_v8.2.pt")

    print("Exportando a ONNX optimizado para Jetson...")
    path = model.export(
        format="onnx",
        imgsz=640,          # Usa 320 para mejor rendimiento en Jetson Nano
        half=False,         # ONNX no siempre soporta FP16 bien, mejor dejar en FP32
        simplify=True,      # Simplificar el grafo para mejor compatibilidad
        dynamic=False,      # Tamaño fijo es mejor para TensorRT
        opset=12,           # Opset 12 tiene buena compatibilidad con Jetson
    )
    print(f"ONNX exportado en: {path}")
    print("\n" + "=" * 60)
    print("  SIGUIENTE PASO: Convertir en la Jetson Nano")
    print("=" * 60)
    print("""
  1. Copia el archivo .onnx a la Jetson Nano
  2. En la Jetson Nano, ejecuta:
  
     python3 -c "
     from ultralytics import YOLO
     model = YOLO('model_v8.2.onnx')
     model.export(format='engine', half=True, imgsz=640)
     "
  
  O usa trtexec directamente:
  
     /usr/src/tensorrt/bin/trtexec \\
       --onnx=model_v8.2.onnx \\
       --saveEngine=model_v8.2.engine \\
       --fp16 \\
       --workspace=2048
       
  3. Usa el .engine resultante para inferencia:
  
     model = YOLO('model_v8.2.engine')
     results = model.track(frame, ...)
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--onnx":
        export_to_onnx_for_jetson()
    else:
        print("Uso:")
        print("  python export_tensorrt.py          -> Exportar directo a TensorRT (necesita GPU NVIDIA)")
        print("  python export_tensorrt.py --onnx   -> Exportar a ONNX optimizado para Jetson")
        print()
        
        respuesta = input("¿Tienes GPU NVIDIA en esta PC? (s/n): ").strip().lower()
        if respuesta == "s":
            export_to_tensorrt()
        else:
            export_to_onnx_for_jetson()
