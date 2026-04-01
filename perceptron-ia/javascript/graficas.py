import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from perceptron   import Perceptron, ACTIVACIONES
from casos_prueba import TODOS_LOS_CASOS

# ── Configuración ─────────────────────────────────────────────────────────────
TASA_APRENDIZAJE = 0.1
MAX_ITERACIONES  = 50
SEMILLA          = 42
CARPETA_SALIDA   = "assets"

# Colores y estilos por activación
ESTILOS = {
    "lineal":    {"color": "#E63946", "linestyle": "-",  "marker": "o", "markersize": 3},
    "escalon":   {"color": "#457B9D", "linestyle": "--", "marker": "s", "markersize": 3},
    "sigmoidal": {"color": "#2A9D8F", "linestyle": "-",  "marker": "^", "markersize": 3},
    "relu":      {"color": "#E9C46A", "linestyle": "-.", "marker": "D", "markersize": 3},
    "softmax":   {"color": "#F4A261", "linestyle": ":",  "marker": "v", "markersize": 3},
    "tanh":      {"color": "#6A0572", "linestyle": "-",  "marker": "P", "markersize": 3},
}

os.makedirs(CARPETA_SALIDA, exist_ok=True)


# ── Entrenamiento y recolección de historiales ────────────────────────────────
def entrenar_caso(caso_data: dict) -> dict:
    """
    Entrena el perceptrón con todas las activaciones para un caso dado.
    Retorna un dict {nombre_activacion: historial_mse}.
    """
    historiales = {}
    for activacion in ACTIVACIONES:
        p = Perceptron(
            n_entradas       = len(caso_data["X"][0]),
            tasa_aprendizaje = TASA_APRENDIZAJE,
            activacion       = activacion,
            max_iter         = MAX_ITERACIONES,
            semilla          = SEMILLA,
        )
        p.entrenar(caso_data["X"], caso_data["y"])
        historiales[activacion] = p.historial_error
    return historiales


# ── Gráfica individual por caso ───────────────────────────────────────────────
def graficar_caso(nombre: str, historiales: dict, ax: plt.Axes, titulo_corto: str):
    """
    Dibuja las 6 curvas de convergencia en el Axes dado.
    """
    epocas = list(range(1, MAX_ITERACIONES + 1))

    for activacion, mse_hist in historiales.items():
        est = ESTILOS[activacion]
        ax.plot(
            epocas,
            mse_hist,
            label       = activacion,
            color       = est["color"],
            linestyle   = est["linestyle"],
            marker      = est["marker"],
            markersize  = est["markersize"],
            markevery   = max(1, MAX_ITERACIONES // 10),
            linewidth   = 1.8,
        )

    ax.set_title(titulo_corto, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Época", fontsize=9)
    ax.set_ylabel("MSE", fontsize=9)
    ax.set_xlim(1, MAX_ITERACIONES)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel global (6 casos en una figura) ─────────────────────────────────────
def generar_panel_global(todos_datos: list):
    """
    Genera una figura 2×3 con las curvas de los 6 casos.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Curvas de Convergencia del Perceptrón\n"
        f"(lr={TASA_APRENDIZAJE}, épocas={MAX_ITERACIONES}, semilla={SEMILLA})",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    titulos_cortos = [
        "AND Lógico",
        "OR Lógico",
        "Clasificación Spam",
        "Predicción del Clima",
        "Detección de Fraudes",
        "Riesgo Académico",
    ]

    axes = []
    for i, (caso_data, titulo) in enumerate(zip(todos_datos, titulos_cortos)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        historiales = entrenar_caso(caso_data)
        graficar_caso(caso_data["nombre"], historiales, ax, titulo)
        axes.append(ax)

    # Leyenda global debajo de la figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc            = "lower center",
        ncol           = 6,
        fontsize       = 10,
        frameon        = True,
        title          = "Función de activación",
        title_fontsize = 10,
        bbox_to_anchor = (0.5, 0.01),
    )

    ruta = os.path.join(CARPETA_SALIDA, "curva_convergencia_global.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"  ✓ Panel global guardado → {ruta}")
    return fig


# ── Gráficas individuales (una por caso) ──────────────────────────────────────
def generar_individuales(todos_datos: list):
    """
    Genera y guarda un PNG individual por cada caso de prueba.
    """
    nombres_archivo = [
        "curva_AND", "curva_OR", "curva_Spam",
        "curva_Clima", "curva_Fraudes", "curva_RiesgoAcademico",
    ]

    for caso_data, nombre_arch in zip(todos_datos, nombres_archivo):
        historiales = entrenar_caso(caso_data)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.suptitle(
            f"Convergencia — {caso_data['nombre']}\n"
            f"(lr={TASA_APRENDIZAJE}, épocas={MAX_ITERACIONES})",
            fontsize=12, fontweight="bold"
        )
        graficar_caso(caso_data["nombre"], historiales, ax, caso_data["nombre"])
        ax.legend(
            loc="upper right", fontsize=9,
            title="Activación", title_fontsize=9,
            framealpha=0.85
        )
        fig.tight_layout()

        ruta = os.path.join(CARPETA_SALIDA, f"{nombre_arch}.png")
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"  ✓ Guardado → {ruta}")
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GENERANDO CURVAS DE CONVERGENCIA")
    print("=" * 60)

    # Cargar todos los casos
    todos_datos = [fn() for fn in TODOS_LOS_CASOS]

    print("\n→ Generando gráficas individuales por caso...")
    generar_individuales(todos_datos)

    print("\n→ Generando panel global (6 casos en una figura)...")
    fig_global = generar_panel_global(todos_datos)

    print("\n→ Abriendo ventana interactiva (ciérrala para terminar)...")
    plt.show()

    print("\n✓ Listo. Revisa la carpeta assets/")
    print("=" * 60)


if __name__ == "__main__":
    main()