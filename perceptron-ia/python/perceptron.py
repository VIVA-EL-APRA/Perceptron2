"""
perceptron.py
=============
Implementación de un Perceptrón desde cero en Python.
Sin librerías de Machine Learning externas.

Funciones de activación disponibles:
  - lineal
  - escalon
  - sigmoidal
  - relu
  - softmax
  - tanh
"""

import math
import random


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de activación y sus derivadas
# ──────────────────────────────────────────────────────────────────────────────

def lineal(x):
    """f(x) = x"""
    return x

def d_lineal(x):
    return 1.0


def escalon(x):
    """f(x) = 1 si x >= 0, 0 si x < 0"""
    return 1.0 if x >= 0 else 0.0

def d_escalon(x):
    # Derivada constante (aproximación para regla delta)
    return 1.0


def sigmoidal(x):
    """f(x) = 1 / (1 + e^-x)"""
    x = max(-500.0, min(500.0, x))   # evitar overflow
    return 1.0 / (1.0 + math.exp(-x))

def d_sigmoidal(x):
    s = sigmoidal(x)
    return s * (1.0 - s)


def relu(x):
    """f(x) = max(0, x)"""
    return max(0.0, x)

def d_relu(x):
    return 1.0 if x > 0 else 0.0


def softmax(x):
    """Versión binaria: e^x / (e^x + 1)"""
    x = max(-500.0, min(500.0, x))
    ex = math.exp(x)
    return ex / (ex + 1.0)

def d_softmax(x):
    s = softmax(x)
    return s * (1.0 - s)


def tanh_act(x):
    """f(x) = tanh(x)"""
    return math.tanh(x)

def d_tanh(x):
    return 1.0 - math.tanh(x) ** 2


# Mapa nombre → (función, derivada)
ACTIVACIONES = {
    "lineal":    (lineal,    d_lineal),
    "escalon":   (escalon,   d_escalon),
    "sigmoidal": (sigmoidal, d_sigmoidal),
    "relu":      (relu,      d_relu),
    "softmax":   (softmax,   d_softmax),
    "tanh":      (tanh_act,  d_tanh),
}


# ──────────────────────────────────────────────────────────────────────────────
# Clase Perceptrón
# ──────────────────────────────────────────────────────────────────────────────

class Perceptron:
    """
    Perceptrón de una sola capa con entrenamiento por regla delta.

    Parámetros
    ----------
    n_entradas        : número de entradas (features)
    tasa_aprendizaje  : learning rate (lr), por defecto 0.1
    activacion        : nombre de la función de activación
    max_iter          : número máximo de épocas de entrenamiento
    semilla           : semilla para reproducibilidad
    """

    def __init__(
        self,
        n_entradas: int,
        tasa_aprendizaje: float = 0.1,
        activacion: str = "sigmoidal",
        max_iter: int = 50,
        semilla: int = 42,
    ):
        if activacion not in ACTIVACIONES:
            raise ValueError(
                f"Activación '{activacion}' no reconocida. "
                f"Opciones: {list(ACTIVACIONES.keys())}"
            )

        random.seed(semilla)
        self.pesos  = [random.uniform(-0.5, 0.5) for _ in range(n_entradas)]
        self.sesgo  = random.uniform(-0.5, 0.5)
        self.lr     = tasa_aprendizaje
        self.max_iter = max_iter
        self.activacion_nombre = activacion
        self._fn_act, self._fn_der = ACTIVACIONES[activacion]

        # Historial de error MSE por época
        self.historial_error: list[float] = []

    # ── Predicción ────────────────────────────────────────────────────────────

    def _suma_ponderada(self, entradas: list[float]) -> float:
        return sum(w * x for w, x in zip(self.pesos, entradas)) + self.sesgo

    def predecir_raw(self, entradas: list[float]) -> float:
        """Devuelve la salida continua de la función de activación."""
        return self._fn_act(self._suma_ponderada(entradas))

    def predecir(self, entradas: list[float], umbral: float = 0.5) -> int:
        """Devuelve la clase predicha (0 ó 1) usando el umbral dado."""
        return 1 if self.predecir_raw(entradas) >= umbral else 0

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def entrenar(self, X: list[list[float]], y: list[float]) -> None:
        """
        Entrena el perceptrón usando la regla delta.

        Parámetros
        ----------
        X : lista de vectores de entrada
        y : lista de etiquetas deseadas (0 ó 1)
        """
        self.historial_error = []

        for _ in range(self.max_iter):
            error_total = 0.0

            for entradas, objetivo in zip(X, y):
                z      = self._suma_ponderada(entradas)
                salida = self._fn_act(z)
                error  = objetivo - salida
                delta  = error * self._fn_der(z)

                # Actualización de pesos y sesgo (regla delta)
                self.pesos = [w + self.lr * delta * x
                              for w, x in zip(self.pesos, entradas)]
                self.sesgo += self.lr * delta

                error_total += error ** 2

            mse = error_total / len(X)
            self.historial_error.append(round(mse, 8))

    # ── Evaluación ────────────────────────────────────────────────────────────

    def evaluar(
        self,
        X: list[list[float]],
        y: list[float],
        umbral: float = 0.5,
    ) -> dict:
        """
        Evalúa el perceptrón sobre un conjunto de datos.

        Retorna
        -------
        dict con:
          - precision     : porcentaje de aciertos
          - correctos     : número de predicciones correctas
          - total         : total de muestras
          - error_final   : MSE de la última época
          - detalle       : lista con el resultado de cada muestra
        """
        correctos = 0
        detalle   = []

        for entradas, objetivo in zip(X, y):
            raw  = self.predecir_raw(entradas)
            pred = 1 if raw >= umbral else 0
            ok   = pred == int(objetivo)
            if ok:
                correctos += 1
            detalle.append({
                "entrada":    entradas,
                "esperado":   int(objetivo),
                "salida_raw": round(raw, 6),
                "prediccion": pred,
                "correcto":   ok,
            })

        return {
            "precision":   round(correctos / len(X) * 100, 4),
            "correctos":   correctos,
            "total":       len(X),
            "error_final": self.historial_error[-1] if self.historial_error else None,
            "detalle":     detalle,
        }

    # ── Representación ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Perceptron("
            f"n_entradas={len(self.pesos)}, "
            f"activacion='{self.activacion_nombre}', "
            f"lr={self.lr}, "
            f"max_iter={self.max_iter})"
        )