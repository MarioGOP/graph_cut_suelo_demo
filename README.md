# Segmentación de suelo con Graph Cut

Comparación de dos métodos para separar **suelo transitable** de **obstáculos** en imágenes de un TurtleBot 4 simulado. Se evalúa en dos mundos, `maze` y `depot`. Todo se reproduce con Python y las imágenes incluidas: **no necesitas ROS, Gazebo ni los rosbags**.

## Cómo correrlo

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Abre `reproducir_resultados.ipynb` en VS Code, selecciona el kernel de `.venv` y usa **Run All**. Si prefieres Jupyter:

```bash
pip install notebook
jupyter notebook reproducir_resultados.ipynb
```

Tarda de 1 a 2 minutos y sobrescribe la carpeta `resultados/`.

## Contenido

```text
graph_cut_suelo_demo/
├── reproducir_resultados.ipynb   # Corre todo desde cero
├── segmentacion.py               # GrabCut y graph cut con PyMaxflow
├── requirements.txt
├── datos/
│   ├── maze/                     # 886 pares
│   │   ├── rgb/                  # Imágenes de 320×240
│   │   └── gt/                   # Ground truth: 1 = suelo, 0 = no suelo
│   └── depot/                    # 527 pares, misma estructura
└── resultados/
    ├── metricas_por_frame.csv
    ├── resumen.csv
    ├── barrido_lambda.csv
    ├── figuras/                  # IoU por frame y mapas de error
    └── videos/                   # comparacion_<mundo> y errores_<mundo>
```

Los archivos `*_h264.mp4` son los más fáciles de compartir por chat. Solo se generan si tienes `ffmpeg`; sin él, quedan los `.mp4` en `mp4v`, que se abren con VLC.

## Resultados

Promedio por frame, procesando a 160×120 con **λ = 20**:

| Métrica | `maze` GrabCut | `maze` **PyMaxflow** | `depot` GrabCut | `depot` **PyMaxflow** |
|---|---|---|---|---|
| Frames | 886 | 886 | 527 | 527 |
| IoU media | 0.538 | **0.956** | 0.762 | **0.989** |
| IoU mediana | 0.533 | **0.977** | 0.884 | **0.990** |
| IoU mínimo | 0.437 | **0.862** | 0.192 | **0.943** |
| Precisión | 0.9999 | **0.9998** | 0.9996 | **0.992** |
| Recall | 0.538 | **0.956** | 0.763 | **0.998** |

### Efecto de λ

Barrido con 1 de cada 5 frames:

| λ | 0 | 0.5 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|
| IoU `maze` | 0.955 | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 | 0.956 | 0.957 |
| IoU `depot` | 0.914 | 0.946 | 0.958 | 0.974 | 0.980 | 0.984 | **0.989** | 0.989 |
| Precisión `depot` | 0.921 | 0.953 | 0.966 | 0.985 | 0.990 | 0.991 | **0.992** | 0.990 |

En `maze` el color ya separa bien las clases. En `depot`, con piso texturizado, **la suavidad sí aporta**: sube el IoU y reduce los obstáculos marcados como suelo.

Las métricas son reproducibles porque el notebook fija la semilla de GrabCut. Los **tiempos sí cambian** según la máquina y su carga; revisa `resultados/resumen.csv` después de correrlo.

## La idea del método

Se minimiza esta energía con un s-t min-cut:

```text
E(L) = Σ D_p(L_p) + λ Σ w_pq [L_p ≠ L_q]
```

- **Término de datos:** `D_p = -log P(color | clase)`, con un histograma BGR de 16 bins por canal, suavizado con σ = 1.
- **Término de suavidad:** `w_pq = exp(-β ‖I_p - I_q‖²)`, con vecindad 4.
- **Semillas automáticas:** un trapecio frente al robot es suelo y el 28 % superior de la imagen es fondo.

La diferencia principal con GrabCut es que **los modelos de color se entrenan solo con las semillas**. GrabCut tiene que asignar la zona incierta a una clase para inicializarse, y eso contamina su modelo.

## Limitaciones

- **Falta `warehouse`** y escenas con obstáculos de color parecido al suelo.
- **λ = 20 se eligió con estos mismos datos**, así que los resultados de `depot` pueden ser algo optimistas. Para reportarlo sin sesgo hace falta evaluarlo en un bag distinto.
- **Quedan falsos positivos en `depot`:** con λ = 20, la precisión es 0.992, pero las bases de algunas columnas todavía se marcan como suelo. Es el error más peligroso para navegar.
- **Las sombras se pierden:** el peor frame de PyMaxflow (IoU 0.862) deja fuera una franja de sombra sobre el piso, porque su color no aparece en la semilla.
- **Las semillas son fijas:** si un obstáculo queda sobre el trapecio, el método lo tomará como suelo.
