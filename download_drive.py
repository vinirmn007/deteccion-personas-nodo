import os
import sys
import subprocess

def install_gdown():
    try:
        import gdown
    except ImportError:
        print("Instalando la librería 'gdown' para la descarga desde Google Drive...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("Librería 'gdown' instalada correctamente.")
        except Exception as e:
            print(f"No se pudo instalar 'gdown' automáticamente. Por favor, instálala manualmente ejecutando: pip install gdown")
            print(f"Detalle del error: {e}")
            sys.exit(1)

def download_folder():
    install_gdown()
    import gdown

    folder_url = "https://drive.google.com/drive/folders/1ycpz1A1IrL5ap5yFRZ-cYvIBHrPsqhUJ?usp=sharing"
    output_dir = "muestras"
    
    # Crear la carpeta de destino si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Se ha creado el directorio de destino: '{output_dir}/'")
    
    print(f"Descargando contenido en la carpeta '{output_dir}/'...")
    try:
        # Intentar descargar usando la URL del folder
        gdown.download_folder(url=folder_url, output=output_dir, quiet=False)
        print("\n¡Descarga finalizada con éxito!")
    except Exception as e:
        print(f"\nHubo un problema descargando usando la URL completa: {e}")
        print("Intentando descargar usando directamente el ID del folder...")
        try:
            folder_id = "1ycpz1A1IrL5ap5yFRZ-cYvIBHrPsqhUJ"
            gdown.download_folder(id=folder_id, output=output_dir, quiet=False)
            print("\n¡Descarga finalizada con éxito usando el ID del folder!")
        except Exception as e_id:
            print(f"\nError al intentar descargar con el ID del folder: {e_id}")
            print("\nSugerencias de resolución:")
            print("1. Verifica que tienes conexión a internet.")
            print("2. Intenta ejecutar: pip install --upgrade gdown")
            print("3. Google Drive a veces bloquea descargas automatizadas si se excede la cuota de descargas públicas. Inténtalo de nuevo más tarde.")

if __name__ == "__main__":
    download_folder()
