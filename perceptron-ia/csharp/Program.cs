using System;
using System.Collections.Generic;

namespace PerceptronCS
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("╔══════════════════════════════════════════════════╗");
            Console.WriteLine("║   Perceptrón — C#  (6 activaciones, 6 casos)    ║");
            Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

            // Activaciones a probar
            string[] activaciones = { "linear", "escalon", "sigmoidal", "relu", "softmax", "tanh" };

            // Casos de prueba
            var casos = new List<(double[][] X, int[] y, string nombre)>
            {
                CasosPrueba.AND(),
                CasosPrueba.OR(),
                CasosPrueba.Spam(),
                CasosPrueba.Clima(),
                CasosPrueba.Fraude(),
                CasosPrueba.RiesgoAcademico(),
            };

            const int EPOCHS = 500;   // > 10 ✔

            foreach (var (X, y, nombre) in casos)
            {
                Console.WriteLine($"\n{'─',50}");
                Console.WriteLine($"  Caso: {nombre}");
                Console.WriteLine($"{'─',50}");

                foreach (string act in activaciones)
                {
                    Console.WriteLine($"\n  Activación: {act.ToUpper()}");

                    var p = new Perceptron(X[0].Length, learningRate: 0.1, activationName: act);
                    p.Train(X, y, EPOCHS);

                    var (acc, mse) = p.Evaluate(X, y);
                    Console.WriteLine($"  → Precisión: {acc * 100:F1}%  |  MSE: {mse:F6}");

                    // Mostrar predicciones individuales
                    Console.Write("  Predicciones: ");
                    foreach (var fila in X)
                        Console.Write($"{p.PredictBinary(fila)} ");
                    Console.WriteLine();
                    Console.Write("  Esperado:     ");
                    foreach (int label in y)
                        Console.Write($"{label} ");
                    Console.WriteLine();
                }
            }

            Console.WriteLine("\n\n✔ Entrenamiento completo. Presiona Enter para salir.");
            Console.ReadLine();
        }
    }
}