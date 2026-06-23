import os

os.environ.pop('LOCAL_RANK', None)
os.environ.pop('RANK', None)
os.environ.pop('WORLD_SIZE', None)

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from roboflow import Roboflow

class MiDatasetCOCO(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        # Aquí iría la lógica usando pycocotools para leer las imágenes y las cajas
        pass

    def __getitem__(self, idx):
        # Devuelve la imagen y un diccionario con las cajas (boxes) y etiquetas (labels)
        pass

    def __len__(self):
        pass

def train_ssd():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    args = parser.parse_args()

    output_dir = f'/gcs/{args.bucket}/entrenamientos_ssd'
    os.makedirs(output_dir, exist_ok=True)

    # 2. Descargar Dataset en formato COCO
    print("Descargando dataset desde Roboflow (Formato COCO)...")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("tu-workspace").project("tu-proyecto")
    dataset = project.version(1).download("coco") 

    # 3. Preparar DataLoader
    # dataset_train = MiDatasetCOCO(root=f"{dataset.location}/train")
    # data_loader = DataLoader(dataset_train, batch_size=8, shuffle=True)

    # 4. Cargar modelo SSD preentrenado
    print("Cargando modelo SSD300 VGG16...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Cargamos el modelo base
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    
    # IMPORTANTE: Reemplazar la "cabeza" del modelo para que coincida con tu número de clases
    # (Recuerda sumar 1 a tus clases, porque SSD considera el "fondo" como la clase 0)
    num_classes = 4 # Ejemplo: 3 clases tuyas + 1 fondo
    # model.head.classification_head.num_classes = num_classes
    
    model.to(device)

    # 5. Configurar Optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 6. El Bucle de Entrenamiento (Lo que YOLO hacía por detrás)
    epochs = 50
    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # for images, targets in data_loader:
        #     images = list(image.to(device) for image in images)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     
        #     loss_dict = model(images, targets)
        #     losses = sum(loss for loss in loss_dict.values())
        #     
        #     optimizer.zero_grad()
        #     losses.backward()
        #     optimizer.step()
        #     
        #     epoch_loss += losses.item()
            
        print(f"Época {epoch+1}/{epochs} finalizada. Pérdida: {epoch_loss}")

    # 7. Guardar los pesos en tu bucket
    ruta_guardado = os.path.join(output_dir, 'ssd_best.pth')
    torch.save(model.state_dict(), ruta_guardado)
    print(f"Modelo guardado exitosamente en: {ruta_guardado}")

if __name__ == '__main__':
    train_ssd()