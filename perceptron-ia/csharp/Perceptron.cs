using System;

namespace PerceptronCS
{
    // Delegado para funciones de activación
    public delegate double ActivationFunction(double x);

    public class Perceptron
    {
        private double[] weights;
        private double bias;
        private double learningRate;
        private ActivationFunction activation;
        private string activationName;

        public Perceptron(int inputSize, double learningRate, string activationName)
        {
            this.learningRate = learningRate;
            this.activationName = activationName;

            Random rng = new Random(42);
            weights = new double[inputSize];
            for (int i = 0; i < inputSize; i++)
                weights[i] = rng.NextDouble() * 0.1;
            bias = rng.NextDouble() * 0.1;

            activation = GetActivation(activationName);
        }

        // ── Funciones de activación ──────────────────────────────────────────
        public static double Linear(double x)      => x;
        public static double Escalon(double x)     => x >= 0 ? 1.0 : 0.0;
        public static double Sigmoidal(double x)   => 1.0 / (1.0 + Math.Exp(-x));
        public static double ReLU(double x)        => Math.Max(0, x);
        public static double TanH(double x)        => Math.Tanh(x);

        // Softmax sobre un solo escalar (se normaliza con e^x / (e^x + e^0))
        public static double Softmax(double x)
        {
            double ex  = Math.Exp(x);
            double e0  = Math.Exp(0);
            return ex / (ex + e0);
        }

        private ActivationFunction GetActivation(string name)
        {
            return name.ToLower() switch
            {
                "linear"    => Linear,
                "escalon"   => Escalon,
                "sigmoidal" => Sigmoidal,
                "relu"      => ReLU,
                "softmax"   => Softmax,
                "tanh"      => TanH,
                _           => throw new ArgumentException($"Activación desconocida: {name}")
            };
        }

        // ── Inferencia ────────────────────────────────────────────────────────
        public double Predict(double[] inputs)
        {
            double net = bias;
            for (int i = 0; i < weights.Length; i++)
                net += weights[i] * inputs[i];
            return activation(net);
        }

        public int PredictBinary(double[] inputs) => Predict(inputs) >= 0.5 ? 1 : 0;

        // ── Entrenamiento ─────────────────────────────────────────────────────
        public void Train(double[][] X, int[] y, int epochs)
        {
            for (int ep = 0; ep < epochs; ep++)
            {
                double totalLoss = 0;
                for (int i = 0; i < X.Length; i++)
                {
                    double output = Predict(X[i]);
                    double error  = y[i] - output;
                    totalLoss    += error * error;

                    bias += learningRate * error;
                    for (int j = 0; j < weights.Length; j++)
                        weights[j] += learningRate * error * X[i][j];
                }
                if ((ep + 1) % 100 == 0)
                    Console.WriteLine($"  Época {ep + 1,4} | MSE: {totalLoss / X.Length:F6}");
            }
        }

        // ── Métricas ──────────────────────────────────────────────────────────
        public (double accuracy, double mse) Evaluate(double[][] X, int[] y)
        {
            int correct = 0;
            double mse  = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double output = Predict(X[i]);
                mse          += Math.Pow(y[i] - output, 2);
                if (PredictBinary(X[i]) == y[i]) correct++;
            }
            return ((double)correct / X.Length, mse / X.Length);
        }

        public string ActivationName => activationName;
    }
}