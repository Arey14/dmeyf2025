#!/bin/bash

# --- Script Principal para Ejecutar el Pipeline de Churn ---

echo "--- Iniciando el Proceso de Predicción de Churn ---"

# 1. Activar el entorno virtual de Pyenv
# Esto asume que 'pyenv' y 'pyenv-virtualenv' están
# correctamente configurados en tu shell (como en la conversación anterior)
echo "Activando el entorno pyenv: zlightgbm"

# 'pyenv activate' puede ser complicado en scripts. Una forma mucho más
# robusta es simplemente exportar esta variable. Pyenv la detectará
# y usará 'zlightgbm' para todos los comandos de python en este script.
export PYENV_VERSION=zlightgbm # <-- CAMBIO 1: Método de activación robusto

# 2. Verifica si la activación funcionó
# Comprobamos si 'pyenv' ahora sabe qué python usar y si
# la palabra 'zlightgbm' está en la ruta de ese python.
if ! pyenv which python | grep -q "$PYENV_VERSION"; then # <-- CAMBIO 2: Verificación correcta
    echo "Error: No se pudo activar el entorno pyenv 'zlightgbm'."
    echo "Verifica que el entorno exista ('pyenv virtualenvs') y que pyenv esté en tu .zshrc/.bashrc."
    exit 1
fi

echo "Entorno activado. Usando Python de: $(pyenv which python)"

# 3. Escanea, lista y selecciona los scripts .py
# (Tu código original aquí es perfecto y no necesita cambios)

echo "Buscando scripts de Python (.py) en el directorio actual..."
readarray -t scripts < <(find . -maxdepth 1 -name "*.py" | sort)

if [ ${#scripts[@]} -eq 0 ]; then
    echo "No se encontraron scripts .py en este directorio."
    exit 0
fi

echo "Scripts disponibles:"
index=0
for script in "${scripts[@]}"; do
    printf "  %2d) %s\n" $((index + 1)) "${script#./}"
    index=$((index + 1))
done

echo "---"
echo "Introduce los NÚMEROS de los scripts que deseas ejecutar, en el orden deseado."
read -p "Orden de ejecución: " selection

if [ -z "$selection" ]; then
    echo "Cancelado. No se ejecutará ningún script."
    exit 0
fi

# 4. Ejecuta los scripts seleccionados en orden
# (Tu código aquí también es perfecto, solo cambié 'python3' por 'python')

echo "--- Iniciando ejecución de la selección ---"
read -ra selected_indices <<< "$selection"

step=1
total_steps=${#selected_indices[@]}

for i in "${selected_indices[@]}"; do
    if ! [[ "$i" =~ ^[0-9]+$ ]]; then
        echo "Advertencia: '$i' no es un número válido. Omitiendo."
        continue
    fi
    
    script_index=$((i - 1))

    if [ -v "scripts[$script_index]" ]; then
        script_to_run="${scripts[$script_index]}"
        script_name="${script_to_run#./}"

        echo ""
        echo "--- (Paso $step/$total_steps) Ejecutando: $script_name ---"
        
        # Al usar 'pyenv', es mejor llamar a 'python'.
        # PYENV_VERSION se asegurará de que este sea el python
        # correcto de tu entorno 'zlightgbm'.
        python "$script_to_run" # <-- Pequeño ajuste (de python3 a python)
        
        exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            echo ""
            echo "*****************************************************"
            echo "Error: El script '$script_name' finalizó con código $exit_code."
            echo "Abortando el resto del pipeline."
            echo "*****************************************************"
            exit 1
        else
            echo "--- Completado: $script_name ---"
        fi
    else
        echo "Advertencia: Índice '$i' fuera de rango. Omitiendo."
    fi
    
    step=$((step + 1))
done

echo ""
echo "--- Proceso completado ---"
