from google.cloud import aiplatform

# --- CONFIGURACIÓN (Cambia estos valores) ---
PROJECT_ID = "project-9cfd392e-d349-480a-bf9"
REGION = "us-central1" # Asegúrate de usar una región que tenga GPUs disponibles
BUCKET_NAME = "modelos_yolo"
ROBOFLOW_API_KEY = "k1lfw5vNH559hQizTZot"
# --------------------------------------------

# Inicializar la conexión con tu proyecto de Google Cloud
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# Definir el trabajo de entrenamiento
job = aiplatform.CustomTrainingJob(
    display_name="entrenamiento-yolo-roboflow",
    script_path="train.py", # Apunta al archivo que creamos en el paso anterior
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest", # Imagen oficial de PyTorch
    requirements=["ultralytics", "roboflow", "huggingface_hub", "Pillow", "python-json-logger"], # Dependencias que Vertex instalará antes de correr tu script
)

print("Enviando trabajo a Vertex AI...")

# Lanzar la máquina virtual
model = job.run(
    machine_type="n1-standard-4", # Máquina base económica
    accelerator_type="NVIDIA_TESLA_T4", # Tipo de GPU
    accelerator_count=1, # Cantidad de GPUs
    args=[
        f"--bucket={BUCKET_NAME}",
        f"--api_key={ROBOFLOW_API_KEY}"
    ],
)