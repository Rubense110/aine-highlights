# AINE: Helldivers 2 Highlights Detection

Este proyecto se centra en el análisis de información no estructurada (AINE) utilizando un enfoque multimodal para la detección automática de los momentos más destacados (highlights) en partidas (gameplays) de Helldivers 2. 

A través de la integración de modelos de visión y lenguaje, el proyecto evalúa los eventos en el video a partir de texto (queries), empleando el modelo base LanguageBind junto con un "adapter" lineal entrenado (Linear Probe) para especializarse en las características de Helldivers 2.

## Estructura del Proyecto

- `data/`: Contiene los datasets (`master_dataset_final.json`), videos de prueba y el adapter ya entrenado ubicado en `data/models/helldivers_adapter.pth`.
- `notebooks/`: Cuadernos interactivos como `complete_languagebind.ipynb` para probar y visualizar las capacidades multimodales y los resultados de inferencia.
- `scripts/`: Scripts como `train_linear_probe.py` utilizados para entrenar el adapter lineal sobre los embeddings congelados de LanguageBind.
- `third_party/`: Dependencias externas o submódulos, en concreto se necesita el repositorio de LanguageBind.

El proyecto ha sido realizado en un entorno **local** con una GPU **NVIDIA RTX 5070ti** de **16 GB de VRAM** sobre visual studio code. Para los notebooks se ha trabajado también en vs code usando la extensión de jupyter.


## Configuración del Entorno

La ruta recomendada de reproducibilidad usa `uv`, con perfiles separados para evaluar artefactos ya precalculados o ejecutar el pipeline completo. El proyecto se ha desarrollado trabajando con notebooks de Jupyter dentro de VS Code, pero los comandos con `uv` deberían funcionar igualmente desde terminal.

Para este proyecto se recomienda **no activar manualmente el entorno virtual**. `uv sync` crea y mantiene `.venv`, y `uv run ...` garantiza que cada comando se ejecuta con ese entorno. Esto evita errores habituales en notebooks, donde es fácil lanzar Jupyter con un Python distinto al que tiene PyTorch/CUDA instalado.

### 1. Clonar el repositorio

Linux/macOS:

```bash
git clone <URL_DEL_REPOSITORIO>
cd aine-highlights
```

Windows PowerShell:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd aine-highlights
```

### 2. Instalar uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Si `uv --version` sigue diciendo `command not found`, añade `uv` al `PATH` de forma permanente:

```bash
echo 'source "$HOME/.local/bin/env"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

Si estás usando la terminal integrada de VS Code instalado como Snap, el instalador puede dejar `uv` dentro de una ruta tipo `$HOME/snap/code/<version>/.local/bin`. En ese caso, usa la ruta que te muestre el instalador. Por ejemplo:

```bash
source "$HOME/snap/code/226/.local/bin/env"
uv --version
```

Para dejarlo permanente en ese caso:

```bash
echo 'source "$HOME/snap/code/226/.local/bin/env"' >> ~/.bashrc
source ~/.bashrc
uv --version
```

La ruta con `snap/code/<version>` puede cambiar cuando VS Code se actualice. Si ocurre, reinstala `uv` desde una terminal normal de Ubuntu o actualiza esa línea en `~/.bashrc`.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

Si PowerShell no encuentra `uv`, cierra y abre la terminal. Si aún no aparece, añade la ruta habitual al `Path` de usuario:

```powershell
$env:Path = "$HOME\.local\bin;$env:Path"
uv --version
```

### 3. Instalar dependencias del proyecto

Para evaluar los resultados ya precalculados, usando `notebooks/pretrained_laguagebind.ipynb`:

Linux/macOS:

```bash
uv sync --group pretrained --group notebook
uv run python scripts/check_environment.py --mode pretrained
```

Windows PowerShell:

```powershell
uv sync --group pretrained --group notebook
uv run python scripts/check_environment.py --mode pretrained
```

Para reproducir el pipeline completo, usando `notebooks/complete_languagebind.ipynb`:

Linux/macOS:

```bash
uv sync --group full --group notebook
uv run python scripts/check_environment.py --mode full
```

Windows PowerShell:

```powershell
uv sync --group full --group notebook
uv run python scripts/check_environment.py --mode full
```

Si aun así quieres activar `.venv` manualmente, se puede hacer, pero no es necesario:

Linux/Mac:
```bash
source .venv/bin/activate
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

`requirements.txt` se mantiene como referencia/fallback, pero la configuración reproducible principal vive en `pyproject.toml` y `uv.lock`. PyTorch está fijado a builds CUDA `cu128` en `pyproject.toml`; si tu máquina necesita otra variante de CUDA, cambia el índice de PyTorch antes de ejecutar `uv sync`.

### 4. Clonar LanguageBind

El proyecto requiere el código fuente de LanguageBind en la carpeta `third_party`. Debes clonar su repositorio oficial en este directorio:

Linux/macOS:

```bash
git clone https://github.com/PKU-YuanGroup/LanguageBind third_party/LanguageBind
```

Windows PowerShell:

```powershell
git clone https://github.com/PKU-YuanGroup/LanguageBind third_party/LanguageBind
```

## Uso

Una vez configurado el entorno, dispones de todos los datos necesarios para comenzar el análisis.
Aquí se indica como explorar notebooks con jupyter, pero como mencionamos antes, nosotros trabajamos desde vscode. 

### Inferencia y evaluación reproducible
Para probar la evaluación con embeddings, adapter y ground truth ya precalculados:

Linux/macOS:

```bash
uv run jupyter lab notebooks/pretrained_laguagebind.ipynb
```

Windows PowerShell:

```powershell
uv run jupyter lab notebooks/pretrained_laguagebind.ipynb
```

Este notebook compara **fine-tuned** frente a **zero-shot** y calcula **Recall@K** y **MRR**. Es la opción recomendada si solo quieres comprobar resultados sin regenerar datasets, embeddings ni modelo.

También puedes lanzar la evaluación directamente por consola:

Linux/macOS:

```bash
uv run python scripts/evaluate_model.py \
  --index-dir indexes/proxy_720p30_w15_s5 \
  --mode manual \
  --ground-truth data/ground_truth/ground_truth_dataset.json \
  --adapter data/models/helldivers_adapter.pth
```

Windows PowerShell:

```powershell
uv run python scripts/evaluate_model.py `
  --index-dir indexes/proxy_720p30_w15_s5 `
  --mode manual `
  --ground-truth data/ground_truth/ground_truth_dataset.json `
  --adapter data/models/helldivers_adapter.pth
```

Para comparar contra LanguageBind sin adaptar:

Linux/macOS:

```bash
uv run python scripts/evaluate_model.py \
  --index-dir indexes/proxy_720p30_w15_s5 \
  --mode manual \
  --ground-truth data/ground_truth/ground_truth_dataset.json \
  --adapter missing_zero_shot_adapter.pth
```

Windows PowerShell:

```powershell
uv run python scripts/evaluate_model.py `
  --index-dir indexes/proxy_720p30_w15_s5 `
  --mode manual `
  --ground-truth data/ground_truth/ground_truth_dataset.json `
  --adapter missing_zero_shot_adapter.pth
```

### Pipeline completo con Jupyter
Todo el pipeline, desde la creación de los datasets hasta el entrenamiento y la inferencia final, se puede ejecutar y visualizar paso a paso en:

Linux/macOS:

```bash
uv run jupyter lab notebooks/complete_languagebind.ipynb
```

Windows PowerShell:

```powershell
uv run jupyter lab notebooks/complete_languagebind.ipynb
```

Aunque se abre con `jupyter lab`, el flujo usado durante el desarrollo ha sido ejecutar esos notebooks desde VS Code con el kernel del entorno creado por `uv`.

### Reentrenar el Adapter Lineal
Si cuentas con nuevos datos o quieres ajustar los pesos desde cero, puedes utilizar el script de entrenamiento de la siguiente manera:

Linux/macOS:

```bash
uv run python scripts/train_linear_probe.py --dataset data/master_dataset_final.json --video data/proxy_720p30.mp4 --output data/models/helldivers_adapter.pth
```

Windows PowerShell:

```powershell
uv run python scripts/train_linear_probe.py --dataset data/master_dataset_final.json --video data/proxy_720p30.mp4 --output data/models/helldivers_adapter.pth
```
>(Los embeddings de video y texto se procesarán usando tu GPU).

---
**Nota:** El adapter de Helldivers 2 actual ya se encuentra alojado y configurado para funcionar leyendo de la ruta `data/models/helldivers_adapter.pth`.
