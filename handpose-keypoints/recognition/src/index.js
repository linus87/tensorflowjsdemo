// Tiny TFJS train / predict example.
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const wieghts = tf.tensor([[2.6339848 , -0.7821255, -0.9852441, 0.0480192  , -0.3466824 , -1.9451324, -0.2665296, 0.4633622 , 0.5863307 , 1.4673749 ],
  [1.9798787 , -3.6038599, -4.460216 , 0.2952887  , -10.3301592, 2.3445635 , -0.143085 , 0.92823   , 2.5617113 , 2.1121659 ],
  [1.9838854 , -0.3016337, -0.5679148, -4.4008055 , -2.9695716 , 0.0452375 , 0.2540358 , 0.3761422 , 0.0191029 , 0.3871876 ],
  [-5.0452185, 0.655054  , 0.7981847 , 0.4678645  , 0.6775135  , -0.3385722, -0.3506645, -1.8684748, -1.1486729, 3.8903823 ],
  [-8.430047 , 1.0430573 , 0.1428999 , -12.6395006, 0.5788171  , -0.1291275, -8.076951 , 1.1151588 , 0.273112  , 0.7584921 ],
  [-0.3173599, 0.5262296 , -0.0265811, -0.4357024 , -0.1919112 , -2.4167848, 0.0672482 , 0.8349618 , -0.6821571, 1.6574062 ],
  [-0.0781015, 0.9455061 , 0.3056812 , 0.8599624  , 0.1987738  , -0.3718994, -4.5338659, -0.1285613, -0.7152275, -2.9902966],
  [-3.3314159, -4.2465887, 4.936142  , 3.0950744  , 0.9306732  , -0.0423123, -3.0542874, 3.0736477 , -2.4638669, -1.128652 ],
  [0.9750291 , 0.3274792 , -0.0958808, 0.118009   , 0.4248115  , -1.2034907, 1.4910746 , 1.3296365 , -0.7900974, -2.8479481],
  [-1.4355721, 0.406817  , 0.7897013 , 1.3567253  , 0.705112   , 0.0477447 , -3.2462018, -0.7236025, 0.5025509 , -5.1215072],
  [1.6338153 , -2.3408177, -6.9712372, 3.397702   , 2.5956841  , 3.6143422 , -1.8373487, -0.3265107, -1.8837547, 1.1698129 ],
  [1.8685191 , 0.5671845 , 0.5649589 , 0.3055698  , 0.1246265  , 0.7307346 , 1.4903445 , -2.0175452, 0.1810325 , -1.7679509],
  [-4.8163428, 0.1706621 , 0.3639418 , 0.5388684  , 1.2080667  , 0.6291083 , 3.2750857 , -3.0049524, 0.7873013 , -3.3101127],
  [-0.9758605, -1.1150088, 1.296903  , 1.4453139  , 1.4802957  , 2.7682326 , 6.4073501 , -2.96071  , -4.3682613, -0.7844887],
  [1.6052431 , 0.0245548 , -1.1945288, -0.017442  , 0.2870344  , 0.3435654 , 1.4639882 , -1.0085355, -0.3355551, -1.1747116]]
);
const bias = tf.tensor([2.6990962, 0.1053372, -0.4259273, -0.4504739, 0.0568584, -2.4289062, -0.7209694, 0.1070402, -0.5185959, 1.0869477]
);
model.add(tf.layers.dense({weights: [wieghts, bias], units: 10, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

model.summary();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

function toTensorFromPalm(positions) {
  return positions.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [[currentValue[0] - previous[0], currentValue[1] - previous[1], currentValue[2] - previous[2]]];
    else return accumulator.concat([[currentValue[0] - previous[0], currentValue[1] - previous[1], currentValue[2] - previous[2]]]);
  })
}

function convertAnnotationsIntoVector(annotations) {
  let xs = [];
  const palmBase = annotations.palmBase;
  xs.push(toTensorFromPalm(palmBase.concat(annotations.thumb)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.indexFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.middleFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.ringFinger)));
  xs.push(toTensorFromPalm(palmBase.concat(annotations.pinky)));
  
  return tf.tensor(xs);
}

function convertVectorsIntoAngles(annotations) {
  const fingerTensors = convertAnnotationsIntoVector(annotations);
  fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));
  console.log(fingerSegmentVectorsDot);

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((finger, fingerIndex) => finger.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][segmentIndex] / fingerSegmentLengths[fingerIndex][segmentIndex+1]) );
  console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
}

// Train the model
async function trainModel() {
    // Generate some synthetic data for training
  const fit_landmarks_response = await fetch('/test_data/fit/annotations.json');
  const one_landmarks_response = await fetch('/test_data/one/annotations.json');
  const two_landmarks_response = await fetch('/test_data/two/annotations.json');
  const three_landmarks_response = await fetch('/test_data/three/annotations.json');
  const four_landmarks_response = await fetch('/test_data/four/annotations.json');
  const five_landmarks_response = await fetch('/test_data/five/annotations.json');
  const six_landmarks_response = await fetch('/test_data/six/annotations.json');
  const seven_landmarks_response = await fetch('/test_data/seven/annotations.json');
  const eight_landmarks_response = await fetch('/test_data/eight/annotations.json');
  const nine_landmarks_response = await fetch('/test_data/nine/annotations.json');

  const fit_landmarks = await fit_landmarks_response.json();
  const one_landmarks = await one_landmarks_response.json();
  const two_landmarks = await two_landmarks_response.json();
  const three_landmarks = await three_landmarks_response.json();
  const four_landmarks = await four_landmarks_response.json();
  const five_landmarks = await five_landmarks_response.json();
  const six_landmarks = await six_landmarks_response.json();
  const seven_landmarks = await seven_landmarks_response.json();
  const eight_landmarks = await eight_landmarks_response.json();
  const nine_landmarks = await nine_landmarks_response.json();

  const fit_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: convertVectorsIntoAngles(annotations), ys: tf.oneHot([9], 10) };});

  const landmarksDataset = fit_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
    .concatenate(three_landmarks_dataset).concatenate(four_landmarks_dataset).concatenate(five_landmarks_dataset)
    .concatenate(six_landmarks_dataset).concatenate(seven_landmarks_dataset).concatenate(eight_landmarks_dataset).concatenate(nine_landmarks_dataset).batch(1);

  let handposesData = await landmarksDataset.toArray();
  console.log(handposesData);

  let xs = [];
  let ys = [];
  handposesData.forEach((e) => {
    xs.push(e.xs.squeeze());
    ys.push(e.ys.squeeze());
  });

  const x = tf.stack(xs);
  const y = tf.stack(ys);
  x.print();
  y.print();

  const history = await model.fit(x, y, {
    batchSize: 1,
    epochs: 100, // Number of iterations over the entire dataset,
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();