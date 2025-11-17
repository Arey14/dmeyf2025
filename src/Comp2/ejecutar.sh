#!/bin/bash

# --- Script Principal para Ejecutar el Pipeline de Churn ---
# Este script activa el entorno virtual de Python y luego
# permite seleccionar qué scripts ejecutar y en qué orden.

echo "--- Iniciando el Proceso de Predicción de Churn ---"

# 1. Define la ruta a tu entorno virtual
# Asegúrate de que esta ruta sea correcta para tu sistema.
VENV_PATH="$HOME/.venv/.venv-py311/bin/activate"

# 2. Verifica si el entorno virtual existe y actívalo
if [ -f "$VENV_PATH" ]; then
    echo "Activando el entorno virtual en: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "Error: Entorno virtual no encontrado en '$VENV_PATH'."
    echo "Por favor, ajusta la variable VENV_PATH en este script."
    exit 1
fi

# 3. Escanea, lista y selecciona los scripts .py

echo "Buscando scripts de Python (.py) en el directorio actual..."

# Guardamos los scripts encontrados en un array (ordenados)
# 'readarray -t' lee líneas de entrada en un array.
# 'find . -maxdepth 1 -name "*.py"' busca solo en el dir actual.
readarray -t scripts < <(find . -maxdepth 1 -name "*.py" | sort)

# Verificamos si se encontraron scripts
if [ ${#scripts[@]} -eq 0 ]; then
    echo "No se encontraron scripts .py en este directorio."
    exit 0
fi

# Mostramos la lista numerada al usuario
echo "Scripts disponibles:"
index=0
for script in "${scripts[@]}"; do
    # 'printf' nos da un formato alineado (ej: " 1)" vs "10)")
    # '${script#./}' quita el prefijo "./" del nombre
    printf "  %2d) %s\n" $((index + 1)) "${script#./}"
    index=$((index + 1))
done

echo "---"
echo "Introduce los NÚMEROS de los scripts que deseas ejecutar, en el orden deseado."
echo "Ejemplo: 3 1 2 (ejecuta el 3ro, luego el 1ro, luego el 2do)"
echo "Presiona ENTER sin nada para cancelar."
read -p "Orden de ejecución: " selection

# Si el usuario solo presiona ENTER, salimos.
if [ -z "$selection" ]; then
    echo "Cancelado. No se ejecutará ningún script."
    exit 0
fi

# 4. Ejecuta los scripts seleccionados en orden

echo "--- Iniciando ejecución de la selección ---"

# Convierte la entrada del usuario (ej: "3 1 2") en un array (selected_indices)
read -ra selected_indices <<< "$selection"

step=1
total_steps=${#selected_indices[@]}

for i in "${selected_indices[@]}"; do
    # Valida que sea un número
    if ! [[ "$i" =~ ^[0-9]+$ ]]; then
        echo "Advertencia: '$i' no es un número válido. Omitiendo."
        continue
    fi

    # Convierte el índice 1-based (usuario) a 0-based (array)
    script_index=$((i - 1))

    # Valida que el índice exista en nuestro array de scripts
    if [ -v "scripts[$script_index]" ]; then
        script_to_run="${scripts[$script_index]}"
        script_name="${script_to_run#./}" # Nombre limpio para mostrar

        echo ""
        echo "--- (Paso $step/$total_steps) Ejecutando: $script_name ---"
        
        python3 "$script_to_run"
        
        # Captura el código de salida del script de Python
        exit_code=$?
        
        # Si el script falla (código de salida != 0), abortamos el pipeline
        if [ $exit_code -ne 0 ]; then
            echo ""
            echo "*****************************************************"
            echo "Error: El script '$script_name' finalizó con código $exit_code."
            echo "Abortando el resto del pipeline."
            echo "*****************************************************"
            exit 1 # Termina el script de bash
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
