public class CasosPrueba {

    // ── Caso 1: AND lógico ────────────────────────────────────────────────────
    public static Object[] AND() {
        double[][] X = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        int[] y = {0, 0, 0, 1};
        return new Object[]{X, y, "AND Lógico"};
    }

    // ── Caso 2: OR lógico ─────────────────────────────────────────────────────
    public static Object[] OR() {
        double[][] X = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        int[] y = {0, 1, 1, 1};
        return new Object[]{X, y, "OR Lógico"};
    }

    // ── Caso 3: Spam / No-Spam ────────────────────────────────────────────────
    // Features: [frecuencia_palabras_clave, num_links, mayusculas_exceso]
    public static Object[] spam() {
        double[][] X = {
            {0.9, 0.8, 1.0},
            {0.1, 0.0, 0.0},
            {0.7, 0.9, 0.8},
            {0.2, 0.1, 0.0},
            {0.8, 0.7, 0.9},
            {0.0, 0.0, 0.1},
            {0.6, 0.8, 0.7},
            {0.1, 0.2, 0.0},
            {0.95, 0.9, 1.0},
            {0.05, 0.0, 0.0},
        };
        int[] y = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        return new Object[]{X, y, "Clasificación Spam"};
    }

    // ── Caso 4: Predicción del clima ──────────────────────────────────────────
    // Features: [humedad, nubosidad, presion_baja]  → 1 = lluvia
    public static Object[] clima() {
        double[][] X = {
            {0.9,  0.8,  1.0},
            {0.3,  0.2,  0.0},
            {0.8,  0.9,  0.8},
            {0.2,  0.3,  0.1},
            {0.85, 0.75, 0.9},
            {0.4,  0.1,  0.2},
            {0.7,  0.8,  0.7},
            {0.1,  0.2,  0.0},
            {0.95, 0.9,  1.0},
            {0.25, 0.15, 0.1},
        };
        int[] y = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        return new Object[]{X, y, "Predicción del Clima"};
    }

    // ── Caso 5: Detección de fraudes ──────────────────────────────────────────
    // Features: [monto_inusual, ubicacion_extraña, hora_inusual]
    public static Object[] fraude() {
        double[][] X = {
            {1.0,  1.0, 1.0},
            {0.1,  0.0, 0.2},
            {0.9,  0.8, 0.9},
            {0.2,  0.1, 0.0},
            {0.8,  0.9, 0.7},
            {0.0,  0.2, 0.1},
            {0.7,  0.8, 0.8},
            {0.1,  0.0, 0.0},
            {0.95, 1.0, 0.9},
            {0.05, 0.1, 0.0},
        };
        int[] y = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        return new Object[]{X, y, "Detección de Fraudes"};
    }

    // ── Caso 6: Riesgo académico ──────────────────────────────────────────────
    // Features: [asistencia_baja, promedio_bajo, tareas_incompletas]
    public static Object[] riesgoAcademico() {
        double[][] X = {
            {0.8,  0.9,  0.8},
            {0.1,  0.2,  0.1},
            {0.9,  0.8,  0.9},
            {0.2,  0.1,  0.0},
            {0.7,  0.9,  0.7},
            {0.0,  0.1,  0.2},
            {0.8,  0.7,  0.8},
            {0.1,  0.0,  0.1},
            {0.9,  1.0,  0.9},
            {0.05, 0.1,  0.0},
        };
        int[] y = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
        return new Object[]{X, y, "Riesgo Académico"};
    }
}