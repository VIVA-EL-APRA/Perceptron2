"""
main.py
=======
Punto de entrada del perceptrón.

Entrena y evalúa el perceptrón en los 6 casos de prueba
con las 6 funciones de activación disponibles.

Uso:
    python main.py
"""

from perceptron   import Perceptron, ACTIVACIONES
from casos_prueba import TODOS_LOS_CASOS

# ──────────────────────────────────────────────────────────────────────────────
# Configuración global
# ──────────────────────────────────────────────────────────────────────────────

TASA_APRENDIZAJE = 0.1
MAX_ITERACIONES  = 50        # > 10 según requerimientos
UMBRAL           = 0.5
SEMILLA          = 42


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ──────────────────────────────────────────────────────────────────────────────

def separador(char="═", ancho=68):
    print(char * ancho)

def encabezado(texto: str):
    separador()
    print(f"  {texto}")
    separador()

def sub_separador(ancho=68):
    print("─" * ancho)

def imprimir_detalle(resultado: dict):
    """Imprime el detalle muestra a muestra de un caso."""
    for fila in resultado["detalle"]:
        entrada_str = str(fila["entrada"])
        marca = "✓" if fila["correcto"] else "✗"
        print(
            f"    Entrada {entrada_str:<12} "
            f"Esperado={fila['esperado']}  "
            f"Raw={fila['salida_raw']:.4f}  "
            f"Pred={fila['prediccion']}  {marca}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_caso(caso_data: dict, activacion: str) -> dict:
    """
    Entrena y evalúa el perceptrón en un caso con una activación dada.

    Retorna un dict con los resultados del entrenamiento y evaluación.
    """
    X = caso_data["X"]
    y = caso_data["y"]

    perceptron = Perceptron(
        n_entradas        = len(X[0]),
        tasa_aprendizaje  = TASA_APRENDIZAJE,
        activacion        = activacion,
        max_iter          = MAX_ITERACIONES,
        semilla           = SEMILLA,
    )

    perceptron.entrenar(X, y)
    resultado = perceptron.evaluar(X, y, umbral=UMBRAL)

    return {
        "caso":          caso_data["nombre"],
        "activacion":    activacion,
        "precision":     resultado["precision"],
        "correctos":     resultado["correctos"],
        "total":         resultado["total"],
        "error_final":   resultado["error_final"],
        "pesos_finales": [round(w, 4) for w in perceptron.pesos],
        "sesgo_final":   round(perceptron.sesgo, 4),
        "historial":     perceptron.historial_error,
        "detalle":       resultado["detalle"],
    }


def main():
    encabezado("PERCEPTRÓN DESDE CERO — PYTHON")
    print(f"  Tasa de aprendizaje : {TASA_APRENDIZAJE}")
    print(f"  Iteraciones         : {MAX_ITERACIONES}")
    print(f"  Semilla             : {SEMILLA}")
    print(f"  Activaciones        : {', '.join(ACTIVACIONES.keys())}")
    print()

    activaciones = list(ACTIVACIONES.keys())
    resumen_global = []

    for caso_fn in TODOS_LOS_CASOS:
        caso_data = caso_fn()
        nombre    = caso_data["nombre"]

        sub_separador()
        print(f"  CASO: {nombre}")
        print(f"  Entradas : {caso_data['entradas']}")
        print(f"  Muestras : {len(caso_data['X'])}")
        sub_separador()

        mejor = {"precision": -1}

        for activacion in activaciones:
            res = ejecutar_caso(caso_data, activacion)

            # Resumen por activación
            print(
                f"  [{activacion:<10}]  "
                f"Precisión: {res['precision']:6.2f}%  |  "
                f"Error MSE: {res['error_final']:.6f}  |  "
                f"Pesos: {res['pesos_finales']}  Sesgo: {res['sesgo_final']}"
            )

            if res["precision"] > mejor["precision"]:
                mejor = res

            resumen_global.append(res)

        # Detalle del mejor resultado en este caso
        print()
        print(f"  >> Mejor activación: [{mejor['activacion']}] "
              f"con {mejor['precision']}% de precisión")
        print(f"  >> Detalle muestra a muestra ({mejor['activacion']}):")
        imprimir_detalle(mejor)
        print()

    # ── Resumen global ────────────────────────────────────────────────────────
    separador()
    print("  RESUMEN GLOBAL")
    separador()
    print(f"  {'Caso':<28} {'Activación':<12} {'Precisión':>10} {'MSE Final':>12}")
    sub_separador()

    casos_vistos = set()
    for res in resumen_global:
        if res["caso"] not in casos_vistos:
            # Mejor resultado de cada caso
            mejores_por_caso = [
                r for r in resumen_global if r["caso"] == res["caso"]
            ]
            mejor = max(mejores_por_caso, key=lambda r: r["precision"])
            print(
                f"  {mejor['caso']:<28} "
                f"{mejor['activacion']:<12} "
                f"{mejor['precision']:>9.2f}% "
                f"{mejor['error_final']:>12.6f}"
            )
            casos_vistos.add(res["caso"])

    separador()
    print("  Entrenamiento completado con éxito.")
    separador()


if __name__ == "__main__":
    main()