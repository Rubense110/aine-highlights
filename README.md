# AINE: Helldivers 2 Highlights Detection

Este proyecto se centra en el análisis de información no estructurada (AINE) utilizando un enfoque multimodal para la detección automática de los momentos más destacados (highlights) en partidas (gameplays) de Helldivers 2. 

A través de la integración de modelos de visión y lenguaje, el proyecto evalúa los eventos en el video a partir de texto (queries), empleando el modelo base LanguageBind junto con un "adapter" lineal entrenado (Linear Probe) para especializarse en las características de Helldivers 2.

## Estructura del Proyecto

- `data/`: Contiene los datasets (`master_dataset_final.json`), videos de prueba y el adapter ya entrenado ubicado en `data/models/helldivers_adapter.pth`.
- `notebooks/`: Cuadernos interactivos como `languagebind_blackwell_poc.ipynb` para probar y visualizar las capacidades multimodales y los resultados de inferencia.
- `scripts/`: Scripts como `train_linear_probe.py` utilizados para entrenar el adapter lineal sobre los embeddings congelados de LanguageBind.
- `third_party/`: Dependencias externas o submódulos, en concreto se necesita el repositorio de LanguageBind.

## Configuración del Entorno

Sigue estos pasos para lanzar y ejecutar el proyecto en un nuevo entorno utilizando los modelos y datos ya pre-entrenados:

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd aine-highlights
```

### 2. Crear un entorno virtual e instalar dependencias

Se recomienda crear un entorno virtual para no interferir con las librerías del sistema:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Clonar LanguageBind

El proyecto requiere el código fuente de LanguageBind en la carpeta `third_party`. Debes clonar su repositorio oficial en este directorio:

```bash
git clone https://github.com/PKU-YuanGroup/LanguageBind third_party/LanguageBind
```

## Uso

Una vez configurado el entorno, dispones de todos los datos necesarios para comenzar a analizar gameplays. 

### Inferencia y Evaluación con Jupyter Notebooks
Todo el pipeline, desde la creación de los datasets hasta el entrenamiento y la inferencia final, se puede ejecutar y visualizar paso a paso directamente desde el notebook interactivo:

```bash
jupyter notebook notebooks/languagebind_blackwell_poc.ipynb
```

**Si solo quieres probar la inferencia con lo que ya está calculado** (sin entrenar ni generar nuevos datasets), asegúrate de que la variable `EMBEDDING_MODE = "load"` esté definida. Ejecuta las dos primeras secciones de configuración ("Imports y rutas" y "Configuración experimental") y luego puedes **saltar directamente a la sección `## GPU y carga de modelos`** para cargar los pesos y hacer consultas de texto sobre el video.

### Reentrenar el Adapter Lineal
Si cuentas con nuevos datos o quieres ajustar los pesos desde cero, puedes utilizar el script de entrenamiento de la siguiente manera:

```bash
python scripts/train_linear_probe.py --dataset data/master_dataset_final.json --video data/proxy_720p30.mp4 --output data/models/helldivers_adapter.pth
```
>(Los embeddings de video y texto se procesarán usando tu GPU).

---
**Nota:** El adapter de Helldivers 2 actual ya se encuentra alojado y configurado para funcionar leyendo de la ruta `data/models/helldivers_adapter.pth`.
