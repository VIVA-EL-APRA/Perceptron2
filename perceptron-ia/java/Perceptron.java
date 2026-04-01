import java.util.Random;

public class Perceptron {

    // ── Interfaz funcional para activaciones ─────────────────────────────────
    @FunctionalInterface
    public interface ActivationFunction {
        double apply(double x);
    }

    // ── Atributos ─────────────────────────────────────────────────────────────
    private double[] weights;
    private double   bias;
    private double   learningRate;
    private ActivationFunction activation;
    private String   activationName;

    // ── Constructor ───────────────────────────────────────────────────────────
    public Perceptron(int inputSize, double learningRate, String activationName) {
        this.learningRate   = learningRate;
        this.activationName = activationName;

        Random rng = new Random(42);
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++)
            weights[i] = rng.nextDouble() * 0.1;
        bias = rng.nextDouble() * 0.1;

        this.activation = getActivation(activationName);
    }

    // ── Funciones de activación estáticas ─────────────────────────────────────
    public static double linear(double x)    { return x; }
    public static double escalon(double x)   { return x >= 0 ? 1.0 : 0.0; }
    public static double sigmoidal(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    public static double relu(double x)      { return Math.max(0, x); }
    public static double tanh(double x)      { return Math.tanh(x); }

    // Softmax escalar: e^x / (e^x + e^0)
    public static double softmax(double x) {
        double ex = Math.exp(x);
        double e0 = Math.exp(0);
        return ex / (ex + e0);
    }

    private ActivationFunction getActivation(String name) {
        return switch (name.toLowerCase()) {
            case "linear"    -> Perceptron::linear;
            case "escalon"   -> Perceptron::escalon;
            case "sigmoidal" -> Perceptron::sigmoidal;
            case "relu"      -> Perceptron::relu;
            case "softmax"   -> Perceptron::softmax;
            case "tanh"      -> Perceptron::tanh;
            default          -> throw new IllegalArgumentException("Activación desconocida: " + name);
        };
    }

    // ── Inferencia ────────────────────────────────────────────────────────────
    public double predict(double[] inputs) {
        double net = bias;
        for (int i = 0; i < weights.length; i++)
            net += weights[i] * inputs[i];
        return activation.apply(net);
    }

    public int predictBinary(double[] inputs) {
        return predict(inputs) >= 0.5 ? 1 : 0;
    }

    // ── Entrenamiento ─────────────────────────────────────────────────────────
    public void train(double[][] X, int[] y, int epochs) {
        for (int ep = 0; ep < epochs; ep++) {
            double totalLoss = 0;
            for (int i = 0; i < X.length; i++) {
                double output = predict(X[i]);
                double error  = y[i] - output;
                totalLoss    += error * error;

                bias += learningRate * error;
                for (int j = 0; j < weights.length; j++)
                    weights[j] += learningRate * error * X[i][j];
            }
            if ((ep + 1) % 100 == 0)
                System.out.printf("    Época %4d | MSE: %.6f%n", ep + 1, totalLoss / X.length);
        }
    }

    // ── Métricas ──────────────────────────────────────────────────────────────
    public double[] evaluate(double[][] X, int[] y) {
        int correct = 0;
        double mse  = 0;
        for (int i = 0; i < X.length; i++) {
            double output = predict(X[i]);
            mse          += Math.pow(y[i] - output, 2);
            if (predictBinary(X[i]) == y[i]) correct++;
        }
        double accuracy = (double) correct / X.length;
        return new double[]{ accuracy, mse / X.length };
    }

    public String getActivationName() { return activationName; }
}