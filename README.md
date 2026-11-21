# Segmentación de suelo con Graph Cut

Comparación de dos métodos para separar **suelo transitable** de **obstáculos** en imágenes de un TurtleBot 4 simulado. Todo se reproduce con Python y las imágenes incluidas: **no necesitas ROS, Gazebo ni el rosbag**.

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

Tarda alrededor de 1.5 minutos y sobrescribe la carpeta `resultados/`.

## Contenido

```text
graph_cut_suelo_demo/
├── reproducir_resultados.ipynb   # Corre todo desde cero
├── segmentacion.py               # GrabCut y graph cut con PyMaxflow
├── requirements.txt
├── datos/maze/
│   ├── rgb/                      # 886 imágenes de 320×240
│   └── gt/                       # Ground truth: 1 = suelo, 0 = no suelo
└── resultados/
    ├── metricas_por_frame.csv
    ├── resumen.csv
    ├── barrido_lambda.csv
    ├── figuras/
    └── videos/
```

Los archivos `*_h264.mp4` son los más fáciles de compartir por chat. Solo se generan si tienes `ffmpeg`; sin él, quedan los `.mp4` en `mp4v`, que se abren con VLC.

## Resultados en el mundo `maze`

Promedio de 886 frames, procesados a 160×120:

| Métrica | GrabCut (`cv2`) | **Graph cut con PyMaxflow** |
|---|---|---|
| IoU media | 0.538 | **0.956** |
| IoU mediana | 0.533 | **0.976** |
| IoU mínimo | 0.437 | **0.860** |
| Precisión | 0.9999 | **0.9998** |
| Recall | 0.538 | **0.956** |

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

- **Solo se evaluó en `maze`**, el escenario fácil: piso liso y muros oscuros. Falta probar pisos con textura, como `depot` y `warehouse`.
- **λ casi no cambia el resultado aquí** (IoU de 0.955 a 0.957 entre λ = 0 y 50), porque el color ya separa bien las clases. Este experimento todavía no demuestra el valor del término de suavidad.
- **Las sombras se pierden:** el peor frame de PyMaxflow (IoU 0.860) deja fuera una franja de sombra sobre el piso, porque su color no aparece en la semilla.
- **Las semillas son fijas:** si un obstáculo queda sobre el trapecio, el método lo tomará como suelo.
