#!/bin/bash

# Definir el nombre del entorno virtual correctamente (sin espacios en la asignación)
ENV_DIR="vision_env"

# Verificar si el entorno virtual ya existe
if [ -d "$ENV_DIR" ]; then
  echo "The environment already exists."
  exit 0
fi

echo "Creating Python environment: $ENV_DIR"
python3 -m venv "$ENV_DIR"

# Activar el entorno virtual para instalar los paquetes necesarios
source "$ENV_DIR/bin/activate"

# Librerías de FLPN
echo "Installing dependencies for $ENV_DIR"
pip install torch torchvision torchaudio numpy matplotlib scikit-learn  
echo "Environment created successfully"

# Añadir el entorno al .gitignore si no está presente
grep -qxF "$ENV_DIR/" .gitignore || echo "$ENV_DIR/" >> .gitignore

