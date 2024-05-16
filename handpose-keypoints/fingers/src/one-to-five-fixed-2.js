// Tiny TFJS train / predict example.
const numFeatures = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5], dtype: 'float32', batchSize: 1}));
const wieghts = tf.tensor([[0.4285955 , -0.4196281, -2.3498352, -4.0158319, -8.9092369, 6.6489644 ],
  [-6.1820574, 9.5308514 , 4.6734819 , -8.0183411, 9.0317717 , 1.7292143 ],
  [-3.2929487, -8.0325375, 8.8502007 , 4.6598635 , -2.5088193, -1.5274763],
  [-1.7355452, -2.8372762, -5.6640782, 3.2940524 , 5.0955324 , 1.7308692 ],
  [-2.4063451, -1.977569 , -5.6118159, 3.7133763 , 4.4479179 , 0.9957244 ]]
);
const bias = tf.tensor([9.1208906, 1.0854325, -2.4544923, -2.1743815, -8.057126, -6.4409657]);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 6, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// 1 means the finger is up, 0 means the finger is down
const fingers = [[0.8, 0.4, 0.4, 0.4, 0.4], [0.75, 1, 0.4, 0.4, 0.4], [0.6, 1, 1, 0.4, 0.4], [0.5, 0.4, 1, 1, 1], [0.4, 1, 1, 1, 1], [1, 1, 1, 1, 1]];
const fingerTensors = tf.tensor(fingers, [6, 5], 'float32');
// const lables = numbers.mod(tf.scalar(2)).toInt();
const labels = tf.tensor([0, 1, 2, 3, 4, 5], [6], 'int32'); 

labels.print(true);

const classNames = ['Fist', 'One', 'Two', 'Three', 'Four', 'Five'];
// Convert to one-hot encoding (even = [1, 0], odd = [0, 1])
const oneHotLabels = tf.oneHot(labels, 6);
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

