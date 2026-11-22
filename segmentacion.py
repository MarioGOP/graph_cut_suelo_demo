
import time
import cv2
import numpy as np
import maxflow


BINS = 16
SIGMA_HIST = 1.0
LAMBDA = 20.0

def suavizar_hist(hist, sigma):
    r = int(3 * sigma + 0.5)
    k = cv2.getGaussianKernel(2 * r + 1, sigma).ravel()
    for eje in range(3):
        hist = sum(k[i + r] * np.roll(hist, i, eje) for i in range(-r, r + 1))
    return hist

def costos_datos(frame, semilla_suelo, semilla_fondo):
    q = frame.astype(np.int32) * BINS // 256
    idx = q[..., 0] * BINS**2 + q[..., 1] * BINS + q[..., 2]

    def modelo(semilla):
        hist = np.bincount(idx[semilla], minlength=BINS**3).astype(float)
        hist = suavizar_hist(hist.reshape(BINS, BINS, BINS), SIGMA_HIST) + 1e-3
        return (hist / hist.sum()).ravel()

    p_suelo, p_fondo = modelo(semilla_suelo), modelo(semilla_fondo)
    D_suelo = -np.log(p_suelo[idx])
    D_fondo = -np.log(p_fondo[idx])
    return D_suelo, D_fondo

def pesos_vecinos(frame):
    img = frame.astype(np.float32)
    dif_h = np.sum((img[:, 1:] - img[:, :-1]) ** 2, axis=2)
    dif_v = np.sum((img[1:, :] - img[:-1, :]) ** 2, axis=2)
    beta = 1 / (2 * np.mean(np.concatenate((dif_h.ravel(), dif_v.ravel()))) + 1e-6)
    w_h = np.exp(-beta * dif_h)
    w_v = np.exp(-beta * dif_v)
    return np.pad(w_h, ((0, 0), (0, 1))), np.pad(w_v, ((0, 1), (0, 0)))


def inicializar_semillas_geometricas(alto, ancho, metodo="GC_PR_BGD"):
    """
    Construye la máscara trimap/quadmap inicial sin intervención humana:
      - cv2.GC_BGD    (0): Obstáculo / fondo seguro (franja superior del horizonte).
      - cv2.GC_FGD    (1): Suelo transitable seguro (trapecio frente al robot).
      - cv2.GC_PR_BGD (2): Zona incierta a resolver mediante Min-Cut / Max-Flow.
      - cv2.GC_PR_FGD (3): Zona probablemente perteneciente al suelo transitable.
    """
    if metodo == "GC_PR_FGD":
        valor = cv2.GC_PR_FGD
    else:
        valor = cv2.GC_PR_BGD

    mascara = np.full((alto, ancho), valor, dtype=np.uint8)

    # Franja superior (primer 28% de la imagen): se asume obstáculo o cielo
    limite_horizonte = int(alto * 0.28)
    mascara[0:limite_horizonte, :] = cv2.GC_BGD

    # Trapecio inferior central: zona inmediatamente frente a las ruedas
    trapecio_suelo = np.array([[
        (int(ancho * 0.28), alto - 1),          # Vértice inferior izquierdo
        (int(ancho * 0.72), alto - 1),          # Vértice inferior derecho
        (int(ancho * 0.62), int(alto * 0.78)),  # Vértice superior derecho
        (int(ancho * 0.38), int(alto * 0.78))   # Vértice superior izquierdo
    ]], dtype=np.int32)

    cv2.fillPoly(mascara, trapecio_suelo, cv2.GC_FGD)
    return mascara, trapecio_suelo

def procesar_graph_cuts(frame, metodo="GC_PR_BGD"):
    """
    Segmenta el suelo transitable usando GrabCut iterativo y calcula
    el centroide navegable y el porcentaje de espacio libre.
    """
    t_inicio = time.time()
    h, w = frame.shape[:2]

    # Paso 1: Generar máscara inicial de semillas (priors geométricos)
    mask, trapecio = inicializar_semillas_geometricas(h, w, metodo)

    # Arreglos internos requeridos por OpenCV para los Modelos de Mezclas Gaussianas (GMM)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Paso 2: Minimización de energía mediante Graph Cuts (iterCount=1 para tiempo real)
    cv2.grabCut(
        img=frame,
        mask=mask,
        rect=None,
        bgdModel=bgd_model,
        fgdModel=fgd_model,
        iterCount=1,
        mode=cv2.GC_INIT_WITH_MASK
    )

    # Convertir etiquetas de GrabCut a máscara binaria (255 = suelo, 0 = obstáculo)
    mascara_suelo = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Paso 3: Postprocesamiento morfológico (Cierre para eliminar agujeros internos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_limpia = cv2.morphologyEx(mascara_suelo, cv2.MORPH_CLOSE, kernel)

    # Conservar únicamente la componente conexa más grande (el camino principal)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_limpia)
    if num_labels > 1:
        # Índice 0 es el fondo negro; buscamos el área máxima a partir del índice 1
        idx_mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mascara_limpia = np.where(labels == idx_mayor, 255, 0).astype(np.uint8)

    # Paso 4: Extracción de métricas de navegación en la región cercana (60% inferior)
    inicio_roi = int(h * 0.40)
    zona_navegable = mascara_limpia[inicio_roi:, :]
    porcentaje_libre = np.count_nonzero(zona_navegable) / zona_navegable.size

    momentos = cv2.moments(zona_navegable)
    if momentos["m00"] > 0:
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"]) + inicio_roi
    else:
        cx, cy = w // 2, h // 2

    tiempo_ms = (time.time() - t_inicio) * 1000.0

    # Paso 5: Construir imagen de visualización para depuración y demo
    vis = frame.copy()
    overlay_verde = np.zeros_like(frame)
    overlay_verde[mascara_limpia == 255] = (0, 220, 0)
    vis = cv2.addWeighted(vis, 0.70, overlay_verde, 0.30, 0)

    # Dibujar el trapecio de semillas (azul), línea central (amarillo) y centroide (rojo)
    cv2.polylines(vis, trapecio, isClosed=True, color=(255, 150, 0), thickness=1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    return vis, mascara_limpia, cx, porcentaje_libre, tiempo_ms

def procesar_graph_cuts_2(frame):
    """
    Versión alternativa de procesar_graph_cuts con PyMaxflow para la optimización de energía.
    """
    t_inicio = time.time()
    h, w = frame.shape[:2]

    # Paso 1: Generar máscara inicial de semillas (priors geométricos)
    mask, trapecio = inicializar_semillas_geometricas(h, w)

    # Semillas para el suelo y el fondo
    semilla_suelo = mask == cv2.GC_FGD
    semilla_fondo = mask == cv2.GC_BGD
    D_suelo, D_fondo = costos_datos(frame, semilla_suelo, semilla_fondo)
    w_h, w_v = pesos_vecinos(frame)

    g = maxflow.Graph[float]()
    nodos = g.add_grid_nodes((h, w))

    derecha = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    abajo = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    g.add_grid_edges(nodos, weights=LAMBDA * w_h, structure=derecha, symmetric=True)
    g.add_grid_edges(nodos, weights=LAMBDA * w_v, structure=abajo, symmetric=True)

    # Fuente = suelo, sumidero = fondo
    cap_fuente = D_fondo.copy()
    cap_sumidero = D_suelo.copy()

    K = 1e6
    cap_fuente[semilla_suelo] = K
    cap_fuente[semilla_fondo] = 0
    cap_sumidero[semilla_suelo] = 0
    cap_sumidero[semilla_fondo] = K
    
    #
    g.add_grid_tedges(nodos, cap_fuente, cap_sumidero)
    g.maxflow()
    mascara_suelo = ~g.get_grid_segments(nodos)
    mascara_suelo = mascara_suelo.astype(np.uint8) * 255

    # Paso 3: Postprocesamiento morfológico (Cierre para eliminar agujeros internos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_limpia = cv2.morphologyEx(mascara_suelo, cv2.MORPH_CLOSE, kernel)

    # Conservar únicamente la componente conexa más grande (el camino principal)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_limpia)
    if num_labels > 1:
        # Índice 0 es el fondo negro; buscamos el área máxima a partir del índice 1
        idx_mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mascara_limpia = np.where(labels == idx_mayor, 255, 0).astype(np.uint8)

    # Paso 4: Extracción de métricas de navegación en la región cercana (60% inferior)
    inicio_roi = int(h * 0.40)
    zona_navegable = mascara_limpia[inicio_roi:, :]
    porcentaje_libre = np.count_nonzero(zona_navegable) / zona_navegable.size

    momentos = cv2.moments(zona_navegable)
    if momentos["m00"] > 0:
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"]) + inicio_roi
    else:
        cx, cy = w // 2, h // 2

    tiempo_ms = (time.time() - t_inicio) * 1000.0

    # Paso 5: Construir imagen de visualización para depuración y demo
    vis = frame.copy()
    overlay_verde = np.zeros_like(frame)
    overlay_verde[mascara_limpia == 255] = (0, 220, 0)
    vis = cv2.addWeighted(vis, 0.70, overlay_verde, 0.30, 0)

    # Dibujar el trapecio de semillas (azul), línea central (amarillo) y centroide (rojo)
    cv2.polylines(vis, trapecio, isClosed=True, color=(255, 150, 0), thickness=1)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

    return vis, mascara_limpia, cx, porcentaje_libre, tiempo_ms

    