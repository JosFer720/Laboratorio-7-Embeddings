# Laboratorio 7: NLP y embeddings

## Requisitos

- Python 3.11 o posterior
- PyTorch con CUDA opcional
- Al menos ocho gigabytes de memoria disponible
- Espacio suficiente para WikiText-103 y los artefactos de entrenamiento

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Pruebas

```powershell
python -m pytest
```

## Ejecución

Abra `laboratorio_7_embeddings.ipynb` y ejecute las celdas en orden. Sin configuración adicional se ejecuta una corrida pequeña para validar la integración.

Para ejecutar WikiText-103 con veinte millones de tokens y dimensiones de 50, 100 y 300, defina la variable antes de iniciar Jupyter.

```powershell
$env:EMBEDDINGS_LAB_FULL = "1"
python -m jupyterlab
```

Los resultados regenerables se guardan en `artifacts/`. Los datos descargados se almacenan en `data/`.
