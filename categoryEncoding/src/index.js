// Tiny TFJS train / predict example.
var numFeatures = 1;

// Define the model architecture
var model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [numFeatures]}));
model.add(tf.layers.categoryEncoding({numTokens: 10, outputMode: "count"}));

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

// Generate some synthetic data for training
// const numbers = tf.range(0, 10, 1); // Generate numbers from 0 to 99
var numbers = tf.rand([10], () => Math.floor(Math.random() * 10), 'int32'); // Generate numbers from 0 to 99
numbers.print();

var input = tf.reshape(numbers, [numFeatures, 10]);
input.print();

model.predict(input).print();