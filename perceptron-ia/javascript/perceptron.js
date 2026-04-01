'use strict';

class Perceptron {
    /**
     * @param {number}   inputSize      – número de entradas
     * @param {number}   learningRate   – tasa de aprendizaje
     * @param {string}   activationName – nombre de la función de activación
     */
    constructor(inputSize, learningRate, activationName) {
        this.learningRate   = learningRate;
        this.activationName = activationName;

        // Inicialización aleatoria reproducible (semilla manual simple)
        let seed = 42;
        const rand = () => {
            seed = (seed * 1664525 + 1013904223) & 0xffffffff;
            return (seed >>> 0) / 0xffffffff;
        };

        this.weights = Array.from({ length: inputSize }, () => rand() * 0.1);
        this.bias    = rand() * 0.1;
        this.activation = this._getActivation(activationName);
    }

    // ── Funciones de activación ──────────────────────────────────────────────
    static linear(x)    { return x; }
    static escalon(x)   { return x >= 0 ? 1 : 0; }
    static sigmoidal(x) { return 1 / (1 + Math.exp(-x)); }
    static relu(x)      { return Math.max(0, x); }
    static tanh(x)      { return Math.tanh(x); }

    // Softmax escalar: e^x / (e^x + e^0)
    static softmax(x) {
        const ex = Math.exp(x);
        const e0 = Math.exp(0);
        return ex / (ex + e0);
    }

    _getActivation(name) {
        const map = {
            linear:    Perceptron.linear,
            escalon:   Perceptron.escalon,
            sigmoidal: Perceptron.sigmoidal,
            relu:      Perceptron.relu,
            softmax:   Perceptron.softmax,
            tanh:      Perceptron.tanh,
        };
        if (!map[name]) throw new Error(`Activación desconocida: ${name}`);
        return map[name];
    }

    // ── Inferencia ────────────────────────────────────────────────────────────
    predict(inputs) {
        const net = inputs.reduce((acc, x, i) => acc + this.weights[i] * x, this.bias);
        return this.activation(net);
    }

    predictBinary(inputs) { return this.predict(inputs) >= 0.5 ? 1 : 0; }

    // ── Entrenamiento ─────────────────────────────────────────────────────────
    train(X, y, epochs) {
        for (let ep = 0; ep < epochs; ep++) {
            let totalLoss = 0;

            for (let i = 0; i < X.length; i++) {
                const output = this.predict(X[i]);
                const error  = y[i] - output;
                totalLoss   += error * error;

                this.bias += this.learningRate * error;
                for (let j = 0; j < this.weights.length; j++)
                    this.weights[j] += this.learningRate * error * X[i][j];
            }

            if ((ep + 1) % 100 === 0)
                console.log(`    Época ${String(ep + 1).padStart(4)} | MSE: ${(totalLoss / X.length).toFixed(6)}`);
        }
    }

    // ── Métricas ──────────────────────────────────────────────────────────────
    evaluate(X, y) {
        let correct = 0;
        let mse     = 0;

        for (let i = 0; i < X.length; i++) {
            const output = this.predict(X[i]);
            mse         += Math.pow(y[i] - output, 2);
            if (this.predictBinary(X[i]) === y[i]) correct++;
        }

        return {
            accuracy: correct / X.length,
            mse:      mse / X.length,
        };
    }
}

module.exports = Perceptron;