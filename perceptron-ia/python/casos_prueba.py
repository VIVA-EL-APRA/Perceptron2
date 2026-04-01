"""
casos_prueba.py
===============
Datos de los 6 casos de clasificación binaria para el perceptrón.

Cada caso devuelve un diccionario con:
  - X         : lista de vectores de entrada (features normalizados 0/1)
  - y         : lista de etiquetas esperadas (0 ó 1)
  - nombre    : nombre descriptivo del caso
  - entradas  : nombres de las variables de entrada
  - descripcion: breve descripción del problema
"""


def caso_and() -> dict:
    """
    AND Lógico
    ----------
    Dos entradas binarias. Salida 1 solo si AMBAS entradas son 1.

    Tabla de verdad:
      A  B  |  A AND B
      0  0  |    0
      0  1  |    0
      1  0  |    0
      1  1  |    1
    """
    return {
        "nombre": "AND Lógico",
        "descripcion": "Compuerta lógica AND — salida 1 solo si A=1 y B=1.",
        "entradas": ["A", "B"],
        "X": [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        "y": [0, 0, 0, 1],
    }


def caso_or() -> dict:
    """
    OR Lógico
    ---------
    Dos entradas binarias. Salida 1 si AL MENOS UNA entrada es 1.

    Tabla de verdad:
      A  B  |  A OR B
      0  0  |    0
      0  1  |    1
      1  0  |    1
      1  1  |    1
    """
    return {
        "nombre": "OR Lógico",
        "descripcion": "Compuerta lógica OR — salida 1 si A=1 o B=1.",
        "entradas": ["A", "B"],
        "X": [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        "y": [0, 1, 1, 1],
    }


def caso_spam() -> dict:
    """
    Clasificación de Correo Spam
    ----------------------------
    Entradas:
      x1 = cantidad_links       (0=pocos, 1=muchos)
      x2 = palabras_clave       (0=no, 1=sí — "oferta", "gratis", "urgente")
      x3 = remitente_desconocido(0=conocido, 1=desconocido)

    Salida:
      1 = SPAM, 0 = NO SPAM
    """
    return {
        "nombre": "Clasificación Spam",
        "descripcion": "Detecta si un correo es spam según links, palabras clave y remitente.",
        "entradas": ["cantidad_links", "palabras_clave", "remitente_desconocido"],
        "X": [
            [0, 0, 0],   # correo limpio
            [0, 0, 1],   # remitente raro, pero sin señales de spam
            [1, 0, 0],   # tiene links, pero remitente conocido
            [0, 1, 0],   # palabras clave → spam
            [1, 1, 0],   # links + palabras → spam
            [1, 1, 1],   # todos los indicadores → spam
            [0, 1, 1],   # palabras + remitente → spam
            [1, 0, 1],   # links + remitente → spam
            [0, 0, 0],   # correo limpio (repetición)
            [1, 1, 1],   # todos los indicadores → spam
            [0, 1, 0],   # palabras clave → spam
            [1, 0, 0],   # solo links → no spam
            [0, 0, 1],   # solo remitente → no spam
            [1, 1, 0],   # links + palabras → spam
        ],
        "y": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    }


def caso_clima() -> dict:
    """
    Predicción del Clima
    --------------------
    Entradas:
      x1 = temperatura_alta  (0=baja, 1=alta)
      x2 = humedad_alta      (0=baja, 1=alta)
      x3 = presion_baja      (0=normal/alta, 1=baja)

    Salida:
      1 = lluvia probable, 0 = sin lluvia
    """
    return {
        "nombre": "Predicción del Clima",
        "descripcion": "Predice lluvia según temperatura, humedad y presión atmosférica.",
        "entradas": ["temperatura_alta", "humedad_alta", "presion_baja"],
        "X": [
            [0, 0, 0],   # frío, seco, presión normal → no llueve
            [0, 1, 0],   # frío, húmedo, presión normal → no llueve
            [1, 0, 0],   # cálido, seco, presión normal → no llueve
            [1, 1, 1],   # cálido, húmedo, presión baja → llueve
            [0, 1, 1],   # frío, húmedo, presión baja → llueve
            [1, 1, 0],   # cálido, húmedo, presión normal → llueve
            [0, 0, 1],   # frío, seco, presión baja → llueve
            [1, 0, 1],   # cálido, seco, presión baja → no llueve
            [0, 0, 0],   # frío, seco, presión normal → no llueve
            [1, 1, 1],   # cálido, húmedo, presión baja → llueve
            [0, 1, 1],   # frío, húmedo, presión baja → llueve
            [1, 0, 0],   # cálido, seco, presión normal → no llueve
            [0, 1, 0],   # frío, húmedo, presión normal → no llueve
            [1, 1, 0],   # cálido, húmedo, presión normal → llueve
        ],
        "y": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    }


def caso_fraude() -> dict:
    """
    Detección de Fraudes
    --------------------
    Entradas:
      x1 = monto_inusual         (0=normal, 1=inusual)
      x2 = ubicacion_extranjera  (0=local, 1=extranjera)
      x3 = hora_inusual          (0=horario normal, 1=madrugada/fin de semana)

    Salida:
      1 = transacción fraudulenta, 0 = legítima
    """
    return {
        "nombre": "Detección de Fraudes",
        "descripcion": "Detecta transacciones fraudulentas por monto, ubicación y horario.",
        "entradas": ["monto_inusual", "ubicacion_extranjera", "hora_inusual"],
        "X": [
            [0, 0, 0],   # todo normal → legítima
            [1, 0, 0],   # monto raro, pero local y horario OK → legítima
            [0, 1, 0],   # ubicación extranjera, resto OK → legítima
            [0, 0, 1],   # hora rara, resto OK → legítima
            [1, 1, 0],   # monto + ubicación → fraude
            [1, 0, 1],   # monto + hora → fraude
            [0, 1, 1],   # ubicación + hora → fraude
            [1, 1, 1],   # todo indica fraude → fraude
            [0, 0, 0],   # todo normal → legítima
            [1, 1, 1],   # todo indica fraude → fraude
            [0, 1, 0],   # solo ubicación → legítima
            [1, 0, 1],   # monto + hora → fraude
            [1, 0, 0],   # solo monto → legítima
            [0, 0, 1],   # solo hora → legítima
        ],
        "y": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    }


def caso_riesgo_academico() -> dict:
    """
    Clasificación de Alumnos con Riesgo Académico
    -----------------------------------------------
    Entradas:
      x1 = asistencia_baja    (0=regular, 1=baja — menos del 70%)
      x2 = promedio_bajo      (0=aprobatorio, 1=bajo — menos de 6.0)
      x3 = entregas_tardias   (0=puntual, 1=más del 50% de tareas tardías)

    Salida:
      1 = alumno en riesgo académico, 0 = sin riesgo
    """
    return {
        "nombre": "Riesgo Académico",
        "descripcion": "Identifica alumnos en riesgo según asistencia, promedio y entregas.",
        "entradas": ["asistencia_baja", "promedio_bajo", "entregas_tardias"],
        "X": [
            [0, 0, 0],   # sin indicadores → sin riesgo
            [1, 0, 0],   # solo asistencia baja → sin riesgo
            [0, 1, 0],   # solo promedio bajo → sin riesgo
            [0, 0, 1],   # solo entregas tardías → sin riesgo
            [1, 1, 0],   # asistencia + promedio → riesgo
            [1, 0, 1],   # asistencia + entregas → riesgo
            [0, 1, 1],   # promedio + entregas → riesgo
            [1, 1, 1],   # todos los indicadores → riesgo alto
            [0, 0, 0],   # sin indicadores → sin riesgo
            [1, 1, 1],   # todos los indicadores → riesgo alto
            [0, 1, 0],   # solo promedio bajo → riesgo (borde)
            [1, 0, 0],   # solo asistencia → sin riesgo
            [1, 1, 0],   # asistencia + promedio → riesgo
            [0, 1, 1],   # promedio + entregas → riesgo
        ],
        "y": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lista de todos los casos disponibles
# ──────────────────────────────────────────────────────────────────────────────

TODOS_LOS_CASOS = [
    caso_and,
    caso_or,
    caso_spam,
    caso_clima,
    caso_fraude,
    caso_riesgo_academico,
]