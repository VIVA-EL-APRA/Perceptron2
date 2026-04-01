'use strict';

const Perceptron  = require('./perceptron');
const casos       = require('./casosPrueba');

console.log('╔══════════════════════════════════════════════════╗');
console.log('║ Perceptrón — JavaScript (6 activaciones, 6 casos)║');
console.log('╚══════════════════════════════════════════════════╝\n');

const ACTIVACIONES = ['linear', 'escalon', 'sigmoidal', 'relu', 'softmax', 'tanh'];
const CASOS        = Object.values(casos);
const EPOCHS       = 500;   // > 10 ✔

for (const { nombre, X, y } of CASOS) {
    console.log('\n──────────────────────────────────────────────────');
    console.log(`  Caso: ${nombre}`);
    console.log('──────────────────────────────────────────────────');

    for (const act of ACTIVACIONES) {
        console.log(`\n  Activación: ${act.toUpperCase()}`);

        const p = new Perceptron(X[0].length, 0.1, act);
        p.train(X, y, EPOCHS);

        const { accuracy, mse } = p.evaluate(X, y);
        console.log(`  → Precisión: ${(accuracy * 100).toFixed(1)}%  |  MSE: ${mse.toFixed(6)}`);

        const preds    = X.map(fila => p.predictBinary(fila)).join(' ');
        const esperado = y.join(' ');
        console.log(`  Predicciones: ${preds}`);
        console.log(`  Esperado:     ${esperado}`);
    }
}

console.log('\n\n✔ Entrenamiento completo.');