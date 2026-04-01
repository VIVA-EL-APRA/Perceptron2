import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║   Perceptrón — Java  (6 activaciones, 6 casos)  ║");
        System.out.println("╚══════════════════════════════════════════════════╝\n");

        String[] activaciones = {"linear", "escalon", "sigmoidal", "relu", "softmax", "tanh"};

        // Todos los casos de prueba
        List<Object[]> casos = Arrays.asList(
            CasosPrueba.AND(),
            CasosPrueba.OR(),
            CasosPrueba.spam(),
            CasosPrueba.clima(),
            CasosPrueba.fraude(),
            CasosPrueba.riesgoAcademico()
        );

        final int EPOCHS = 500;   // > 10 ✔

        for (Object[] caso : casos) {
            double[][] X   = (double[][]) caso[0];
            int[]      y   = (int[])     caso[1];
            String     nom = (String)    caso[2];

            System.out.println("\n──────────────────────────────────────────────────");
            System.out.println("  Caso: " + nom);
            System.out.println("──────────────────────────────────────────────────");

            for (String act : activaciones) {
                System.out.println("\n  Activación: " + act.toUpperCase());

                Perceptron p = new Perceptron(X[0].length, 0.1, act);
                p.train(X, y, EPOCHS);

                double[] metrics = p.evaluate(X, y);
                System.out.printf("  → Precisión: %.1f%%  |  MSE: %.6f%n",
                                  metrics[0] * 100, metrics[1]);

                System.out.print("  Predicciones: ");
                for (double[] fila : X)
                    System.out.print(p.predictBinary(fila) + " ");
                System.out.println();

                System.out.print("  Esperado:     ");
                for (int label : y)
                    System.out.print(label + " ");
                System.out.println();
            }
        }

        System.out.println("\n\n✔ Entrenamiento completo.");
    }
}