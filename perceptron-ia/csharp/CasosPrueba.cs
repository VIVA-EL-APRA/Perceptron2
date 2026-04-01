namespace PerceptronCS
{
    public static class CasosPrueba
    {
        // ── Caso 1: AND lógico ────────────────────────────────────────────────
        public static (double[][] X, int[] y, string nombre) AND()
        {
            double[][] X =
            {
                new[] {0.0, 0.0},
                new[] {0.0, 1.0},
                new[] {1.0, 0.0},
                new[] {1.0, 1.0},
            };
            int[] y = { 0, 0, 0, 1 };
            return (X, y, "AND Lógico");
        }

        // ── Caso 2: OR lógico ─────────────────────────────────────────────────
        public static (double[][] X, int[] y, string nombre) OR()
        {
            double[][] X =
            {
                new[] {0.0, 0.0},
                new[] {0.0, 1.0},
                new[] {1.0, 0.0},
                new[] {1.0, 1.0},
            };
            int[] y = { 0, 1, 1, 1 };
            return (X, y, "OR Lógico");
        }

        // ── Caso 3: Spam / No-Spam ────────────────────────────────────────────
        // Features: [frecuencia_palabras_clave, num_links, tiene_mayusculas_exceso]
        public static (double[][] X, int[] y, string nombre) Spam()
        {
            double[][] X =
            {
                new[] {0.9, 0.8, 1.0},   // spam
                new[] {0.1, 0.0, 0.0},   // no spam
                new[] {0.7, 0.9, 0.8},   // spam
                new[] {0.2, 0.1, 0.0},   // no spam
                new[] {0.8, 0.7, 0.9},   // spam
                new[] {0.0, 0.0, 0.1},   // no spam
                new[] {0.6, 0.8, 0.7},   // spam
                new[] {0.1, 0.2, 0.0},   // no spam
                new[] {0.95, 0.9, 1.0},  // spam
                new[] {0.05, 0.0, 0.0},  // no spam
            };
            int[] y = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            return (X, y, "Clasificación Spam");
        }

        // ── Caso 4: Predicción del clima ──────────────────────────────────────
        // Features: [humedad, nubosidad, presion_baja]  → 1 = lluvia
        public static (double[][] X, int[] y, string nombre) Clima()
        {
            double[][] X =
            {
                new[] {0.9, 0.8, 1.0},   // lluvia
                new[] {0.3, 0.2, 0.0},   // no lluvia
                new[] {0.8, 0.9, 0.8},   // lluvia
                new[] {0.2, 0.3, 0.1},   // no lluvia
                new[] {0.85, 0.75, 0.9}, // lluvia
                new[] {0.4, 0.1, 0.2},   // no lluvia
                new[] {0.7, 0.8, 0.7},   // lluvia
                new[] {0.1, 0.2, 0.0},   // no lluvia
                new[] {0.95, 0.9, 1.0},  // lluvia
                new[] {0.25, 0.15, 0.1}, // no lluvia
            };
            int[] y = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            return (X, y, "Predicción del Clima");
        }

        // ── Caso 5: Detección de fraudes ──────────────────────────────────────
        // Features: [monto_inusual, ubicacion_extraña, hora_inusual]
        public static (double[][] X, int[] y, string nombre) Fraude()
        {
            double[][] X =
            {
                new[] {1.0, 1.0, 1.0},   // fraude
                new[] {0.1, 0.0, 0.2},   // legítimo
                new[] {0.9, 0.8, 0.9},   // fraude
                new[] {0.2, 0.1, 0.0},   // legítimo
                new[] {0.8, 0.9, 0.7},   // fraude
                new[] {0.0, 0.2, 0.1},   // legítimo
                new[] {0.7, 0.8, 0.8},   // fraude
                new[] {0.1, 0.0, 0.0},   // legítimo
                new[] {0.95, 1.0, 0.9},  // fraude
                new[] {0.05, 0.1, 0.0},  // legítimo
            };
            int[] y = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            return (X, y, "Detección de Fraudes");
        }

        // ── Caso 6: Riesgo académico ──────────────────────────────────────────
        // Features: [asistencia_baja, promedio_bajo, tareas_incompletas]
        public static (double[][] X, int[] y, string nombre) RiesgoAcademico()
        {
            double[][] X =
            {
                new[] {0.8, 0.9, 0.8},   // en riesgo
                new[] {0.1, 0.2, 0.1},   // sin riesgo
                new[] {0.9, 0.8, 0.9},   // en riesgo
                new[] {0.2, 0.1, 0.0},   // sin riesgo
                new[] {0.7, 0.9, 0.7},   // en riesgo
                new[] {0.0, 0.1, 0.2},   // sin riesgo
                new[] {0.8, 0.7, 0.8},   // en riesgo
                new[] {0.1, 0.0, 0.1},   // sin riesgo
                new[] {0.9, 1.0, 0.9},   // en riesgo
                new[] {0.05, 0.1, 0.0},  // sin riesgo
            };
            int[] y = { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            return (X, y, "Riesgo Académico");
        }
    }
}