// Tiny TFJS train / predict example.
const numFeatures = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5], dtype: 'float32', batchSize: 1}));
const wieghts = tf.tensor([[1.0883788 , 0.8712656 , -0.9700499, -1.4887093, -1.7502773, 0.7413311, 1.034719  , -0.8100567, 0.7494184 , -1.5656507],
  [-0.6161675, 1.3949009 , 1.9566973 , -1.6268402, 0.5835654 , 0.1054594, -0.9755827, -0.0001793, 0.8707672 , 0.9623067 ],
  [-0.604306 , -1.2760693, 1.7292457 , 1.5961143 , 0.7940789 , 0.1552009, -1.0748837, 1.1436024 , -1.5696244, -1.7751746],
  [0.5617987 , 0.8360516 , -0.8171754, 1.2278987 , 1.071833  , 1.3054754, -1.6929121, -1.4328282, -1.5908208, -1.1369481],
  [-1.0560938, -0.6138406, -0.8500627, 1.1931524 , 1.2723131 , 0.7859063, 0.891067  , -1.3463449, -0.7452984, -1.4209027]]
);
const bias = tf.tensor([0.7241889, -0.5697678, -0.8377176, -0.2259199, -0.826536, -1.1769015, 0.5255417, 1.2907526, 0.6015753, 1.4732449]);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 10, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// 1 means the finger is up, 0 means the finger is down
const fingers = [[0.8, 0.4, 0.4, 0.4, 0.4], [0.75, 1, 0.4, 0.4, 0.4], [0.6, 1, 1, 0.4, 0.4], [0.5, 0.4, 1, 1, 1], [0.4, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [0.5, 0.7, 0.6, 0, 0], [1, 1, 0, 0, 0], [0, 0.5, 0, 0, 0]];
const fingerTensors = tf.tensor(fingers, [10, 5], 'float32');
// const lables = numbers.mod(tf.scalar(2)).toInt();
const labels = tf.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10], 'int32'); 

labels.print(true);

const classNames = ['Fist', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];
// Convert to one-hot encoding (even = [1, 0], odd = [0, 1])
const oneHotLabels = tf.oneHot(labels, 10);
oneHotLabels.print(true);

// Train the model
async function trainModel() {
  const history = await model.fit(fingerTensors, oneHotLabels, {
    epochs: 500, // Number of iterations over the entire dataset,
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

