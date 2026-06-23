import os

os.environ.pop('LOCAL_RANK', None)
os.environ.pop('RANK', None)
os.environ.pop('WORLD_SIZE', None)

import json
import zipfile
import argparse
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def prepare_crowdhuman(output_dir="dataset_crowdhuman"):
    """
    Descarga y convierte el dataset CrowdHuman al formato YOLO.
    """
    print("1. Creando estructura de carpetas...")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
    
    repo_id = "sshao0516/CrowdHuman"
    
    # Nota: Train está dividido en 3 zips (01, 02, 03). Para empezar rápido,
    # el script solo descargará train01 y val. Si quieres el dataset COMPLETO, 
    # agrega "CrowdHuman_train02.zip" y "CrowdHuman_train03.zip" a esta lista.
    files_to_download = [
        "CrowdHuman_val.zip", 
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip", 
        "annotation_val.odgt",
        "annotation_train.odgt"
    ]
    
    paths = {}
    print("2. Descargando archivos desde Hugging Face...")
    for f in files_to_download:
        paths[f] = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
        
    print("3. Extrayendo imágenes (esto puede tomar un momento)...")
    with zipfile.ZipFile(paths["CrowdHuman_val.zip"], 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, "images", "val"))
    with zipfile.ZipFile(paths["CrowdHuman_train01.zip"], 'r') as zip_ref:
        zip_ref.extractall(os.path.join(output_dir, "images", "train"))
        
    print("4. Convirtiendo anotaciones .odgt al formato YOLO (.txt)...")
    def parse_annotations(odgt_path, split):
        with open(odgt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            data = json.loads(line.strip())
            img_id = data["ID"]
            img_path = os.path.join(output_dir, "images", split, f"{img_id}.jpg")
            
            # Validar si la imagen existe (útil porque no estamos bajando los zips train02 y 03)
            if not os.path.exists(img_path):
                continue
                
            # YOLO requiere normalizar coordenadas respecto al ancho y alto de la imagen
            with Image.open(img_path) as img:
                width, height = img.size
                
            label_path = os.path.join(output_dir, "labels", split, f"{img_id}.txt")
            
            with open(label_path, 'w') as label_file:
                for box_data in data["gtboxes"]:
                    # Buscamos la etiqueta "person" y su caja de cuerpo entero "fbox"
                    # Si preferías detectar solo cabezas, cambiarías "fbox" por "hbox"
                    if box_data.get("tag") == "head" and "hbox" in box_data:
                        # CrowdHuman entrega formato: [x_min, y_min, ancho, alto]
                        bx, by, bw, bh = box_data["hbox"]
                        
                        # YOLO usa: [x_center, y_center, ancho, alto] normalizado entre 0 y 1
                        x_center = (bx + (bw / 2.0)) / width
                        y_center = (by + (bh / 2.0)) / height
                        norm_w = bw / width
                        norm_h = bh / height
                        
                        # Limitamos las coordenadas a 1.0 para evitar errores si algo se sale del marco
                        x_center, y_center = min(max(x_center, 0), 1), min(max(y_center, 0), 1)
                        norm_w, norm_h = min(norm_w, 1), min(norm_h, 1)
                        
                        # Clase 0 = person
                        label_file.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        
    parse_annotations(paths["annotation_val.odgt"], "val")
    parse_annotations(paths["annotation_train.odgt"], "train")
    
    print("5. Generando data.yaml...")
    yaml_content = f"""
train: {os.path.abspath(os.path.join(output_dir, 'images', 'train'))}
val: {os.path.abspath(os.path.join(output_dir, 'images', 'val'))}

# Numero de clases
nc: 1

# Nombres de las clases
names: ['person']
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    return yaml_path


def train_yolo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=False) # Ahora no es obligatorio para HF
    parser.add_argument('--api_key', type=str, required=False) # Ahora no es obligatorio para HF
    args = parser.parse_args()

    # Prepara el dataset de HF en la VM de Vertex y obtiene el yaml
    dataset_yaml = prepare_crowdhuman()

    # (Opcional) Si quieres guardar los resultados del entrenamiento en tu Bucket de GCS
    output_dir = f'/gcs/{args.bucket}/entrenamientos_hf' if args.bucket else 'runs/detect'

    print("Cargando modelo base e iniciando entrenamiento...")
    model = YOLO('yolo26n.pt') 

    model.train(
        data=dataset_yaml,
        epochs=100,
        batch=16,
        imgsz=640,
        project=output_dir,
        name='modelo_V7_ch',
        device=0
    )

if __name__ == '__main__':
    train_yolo()
