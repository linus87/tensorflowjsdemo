// Tiny TFJS train / predict example.
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as mobilenet from '@tensorflow-models/mobilenet';

const img = document.getElementById('img');
const version = 1;
const alpha = 1;

let model;

async function run() {
  model = await mobilenet.load({version, alpha});
  console.log('Successfully loaded model');
  const predictBtn = document.getElementById('predict-button');
  predictBtn.removeAttribute("disabled");
  predictBtn.addEventListener('click', clickHandler);
}

run();

async function clickHandler() {
  const predictions = await model.classify(img);
  console.log('Predictions');
  console.log(predictions);

  // Get the logits.
  const logits = model.infer(img);
  console.log('Logits');
  logits.print(true);

  // Get the embedding.
  const embedding = model.infer(img, true);
  console.log('Embedding');
  embedding.print(true);
}