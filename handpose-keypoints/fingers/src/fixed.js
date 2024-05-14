// Tiny TFJS train / predict example.
const numFeatures = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5], dtype: 'int32', batchSize: 1}));
const wieghts = tf.tensor( [[-1.1263417, -1.5367619, -1.9722359, -3.5889871, -7.457149, 7.8942184 ],
    [-7.3158832, 8.0760489 , 3.6547334 , -6.8938174, 7.8840833, 2.8080494 ],
    [-4.1386485, -7.6514487, 7.5131793 , 4.0719628 , -2.219883, -1.1225151],
    [-2.7252691, -2.5927505, -4.8505821, 2.6785848 , 5.2007599, 2.1220226 ],
    [-3.3960693, -1.7330455, -4.7983198, 3.0979087 , 4.5531454, 1.3868775 ]]);
const bias = tf.tensor( [7.5853829, 0.2534243, -2.9552038, -2.4652238, -7.5406632, -5.0107093]);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 6, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// Generate some synthetic data for training
const fingers = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]];
const fingerTensors = tf.tensor(fingers, [6, 5], 'int32');
// const lables = numbers.mod(tf.scalar(2)).toInt();
const labels = tf.tensor([0, 1, 2, 3, 4, 5], [6], 'int32'); // 0 for even, 1 for odd numbers

labels.print(true);

const classNames = ['Fist', 'One', 'Two', 'Three', 'Four', 'Five'];
// Convert to one-hot encoding (even = [1, 0], odd = [0, 1])
const oneHotLabels = tf.oneHot(labels, 6);
oneHotLabels.print(true);

// Train the model
async function trainModel() {
  const history = await model.fit(fingerTensors, oneHotLabels, {
    epochs: 100, // Number of iterations over the entire dataset,
    validationSplit: 1, // 50% of the data will be used for validation
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();

let result = "";
fingers.forEach((finger, index) => {
    const predictions = model.predict(tf.tensor2d(finger, [1, 5]) );
    predictions.print(); // This will output probabilities. You can threshold at 0.5 for binary classification.
  
    result += `<p>Handpose ${classNames[index]}: `;
    predictions.argMax(1).data().then(index => {
        result += `${classNames[index]}:`;
    });
    result += finger;
    result += "</p>";
});
document.getElementById('result').innerHTML = result;

