from marshal import version
import os

os.environ.pop('LOCAL_RANK', None)
os.environ.pop('RANK', None)
os.environ.pop('WORLD_SIZE', None)

import argparse
from ultralytics import YOLO
from roboflow import Roboflow

def train_yolo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True) 
    args = parser.parse_args()

    output_dir = f'/gcs/{args.bucket}/entrenamientos'

    print("Descargando dataset desde Roboflow...")

    rf = Roboflow(api_key="k1lfw5vNH559hQizTZot")
    project = rf.workspace("alexiss-workspace-sentr").project("proyectonodo")
    version = project.version(6)
    dataset = version.download("yolo26")
    
    dataset_yaml = f"{dataset.location}/data.yaml"

    print("Cargando modelo base e iniciando entrenamiento...")
    model = YOLO('yolo26n.pt') 

    model.train(
        data=dataset_yaml,
        epochs=100,
        batch=16,
        imgsz=640,
        project=output_dir,
        name='modelo_v8',
        device=0
    )

if __name__ == '__main__':
    train_yolo()