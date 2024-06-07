// Tiny TFJS train / predict example.
// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 4], dtype: 'float32', batchSize: 1}));
model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const weights = tf.tensor([[1.5817788 , -0.1669038 , -1.2855061 , -0.3926843, -0.2548953, -1.3304799, -0.4840652 , -0.4510006, 0.4670272 , 1.0662682 ],
  [1.701354  , -0.2500079 , -4.002584  , 0.106716  , -8.3002377, 0.1669046 , 0.082515   , 0.3254746 , 0.8817891 , 0.6955653 ],
  [2.1859424 , 0.2679296  , -0.1493268 , -3.2688725, -1.9970329, -0.2913068, 0.0912572  , 0.0239712 , -0.1232208, 0.0003004 ],
  [1.8018247 , -0.6363841 , -1.8504697 , -0.7870008, -3.3872874, -0.7104735, 0.34148    , 0.0315586 , 0.6696062 , 0.6350362 ],
  [-2.5463841, 1.2890463  , 1.0535276  , 0.0814108 , 0.4492211 , -1.1026806, -1.1651013 , -1.5578177, -0.9707677, 1.8530806 ],
  [-9.9699869, 1.1615473  , 0.1296155  , -13.180707, 0.3140518 , -0.7586119, -10.8941221, -0.2735312, 1.0900306 , -0.5247676],
  [0.1943522 , 0.8231057  , -0.3466403 , -0.5948933, -0.2302228, -1.7411318, 0.0034538  , 0.249221  , 0.5410739 , -0.3649305],
  [-0.6598104, 0.7255921  , 0.16032    , -2.2584212, 0.1311153 , -1.0301162, -1.0979056 , -1.0403804, 0.2198458 , 0.2556645 ],
  [-0.7052302, -0.873116  , 1.248158   , 0.9739526 , 0.4328133 , 0.7111865 , -2.1394265 , 1.3128043 , -1.2810245, 0.2847064 ],
  [-4.0104094, -12.6133862, 2.4821341  , 1.5731088 , 0.0968582 , 0.3723301 , -2.7683837 , 1.4376594 , -0.7033057, -2.9085433],
  [-0.3478244, 0.0759465  , 0.5284213  , 0.1314536 , 0.0530684 , -0.5847424, 0.8329626  , 0.9956024 , -0.5356041, 0.0903138 ],
  [0.7612466 , -0.7647948 , 1.0780674  , 0.9896775 , 0.3866164 , -0.1317478, -1.1263878 , 1.5599668 , -2.3218274, 0.8248695 ],
  [-0.6552831, -1.7019352 , 0.8687051  , 1.4310428 , 0.3292782 , 0.7428844 , -1.2649443 , -1.5905865, 1.0138586 , -1.6841209],
  [-0.1413377, -2.2350268 , -11.2070694, 1.8919972 , 1.2123882 , 1.4920254 , -2.8277531 , -1.6567212, -0.1928559, -3.1838593],
  [-0.3387979, 0.3902134  , 0.2039788  , 0.2062245 , 0.2700391 , 0.0570419 , 0.8861928  , 0.6207995 , 0.0741046 , -1.8373051],
  [1.3795205 , -0.6249079 , -1.6851373 , 1.2286162 , 0.9321663 , 0.9161893 , -1.3585653 , 0.0100234 , -1.0035944, 0.7530614 ],
  [-4.0831857, -0.0784373 , 0.0816493  , 1.2010921 , 0.8931215 , 1.4559834 , 3.4860525  , -3.4968038, 0.1978626 , -1.6599416],
  [-1.0469966, -0.0297685 , 2.7172191  , 0.9767167 , 0.9961544 , 1.3621479 , 2.8432     , -1.8712549, -2.4269919, -2.2130511],
  [-0.3563552, -0.293013  , -0.4409429 , 0.5660068 , 0.3545236 , -0.1585491, 1.7104379  , -0.0600608, 0.9285439 , -2.4566383],
  [0.6103426 , 0.0029667  , -0.4918519 , 0.3450321 , 0.7666079 , 0.6163791 , 2.0582728  , -0.3363081, -1.4399294, -0.3189533]]

);
const bias = tf.tensor([2.0379453, 0.0804453, -0.4092534, -0.4515373, -0.1221804, -1.7837509, -0.2543624, -0.336774, 0.016298, 0.7422782]

);
model.add(tf.layers.dense({weights: [weights, bias], units: 10, activation: 'softmax'}));
// model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

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
  // fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  // fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[currentIndex - 1];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));
  // console.log(fingerSegmentVectorsDot);

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((finger, fingerIndex) => finger.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][segmentIndex] / fingerSegmentLengths[fingerIndex][segmentIndex+1]) );
  // console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
}

function distanceToPalm(palmBase, positions) {
  return positions.map(tensor => {
    const x = tensor[0] - palmBase[0];
    const y = tensor[1] - palmBase[1];
    const z = tensor[2] - palmBase[2];
    return Math.sqrt(x*x + y*y + z*z);
   });
}

function convertAnnotationsIntoDistanceFromPalm(annotations) {
  let xs = [];
  const palmBase = annotations.palmBase[0];
  xs.push(distanceToPalm(palmBase, annotations.thumb));
  xs.push(distanceToPalm(palmBase, annotations.indexFinger));
  xs.push(distanceToPalm(palmBase, annotations.middleFinger));
  xs.push(distanceToPalm(palmBase, annotations.ringFinger));
  xs.push(distanceToPalm(palmBase, annotations.pinky));

  let fingerTensors = [];
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.thumb)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.indexFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.middleFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.ringFinger)));
  fingerTensors.push(toTensorFromPalm(annotations.palmBase.concat(annotations.pinky)));

  let fingerLenghs = [];
  fingerTensors.forEach(fingerTensor => {
    let length = 0;
    
    fingerTensor.forEach(tensor => {
      length += Math.sqrt(tensor[0]*tensor[0] + tensor[1]*tensor[1] + tensor[2]*tensor[2]);
    });
    fingerLenghs.push(length);
  });

  xs = xs.map((tensor, index) => [tensor[3] / fingerLenghs[index]]);
  
  return tf.tensor(xs);
}

function mergeFingerAnnotations(annotations) {
  const angles = convertVectorsIntoAngles(annotations);
  angles.print();
  const distances = convertAnnotationsIntoDistanceFromPalm(annotations);
  distances.print();
  angles.concat(distances, 1).print();
  return angles.concat(distances, 1);
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

  const fist_landmarks_dataset = tf.data.array(fit_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([0], 10) };});
  const one_landmarks_dataset = tf.data.array(one_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([1], 10) };});
  const two_landmarks_dataset = tf.data.array(two_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([2], 10) };});
  const three_landmarks_dataset = tf.data.array(three_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([3], 10) };});
  const four_landmarks_dataset = tf.data.array(four_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([4], 10) };});
  const five_landmarks_dataset = tf.data.array(five_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([5], 10) };});
  const six_landmarks_dataset = tf.data.array(six_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([6], 10) };});
  const seven_landmarks_dataset = tf.data.array(seven_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([7], 10) };});
  const eight_landmarks_dataset = tf.data.array(eight_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([8], 10) };});
  const nine_landmarks_dataset = tf.data.array(nine_landmarks).map(annotations => {return {xs: mergeFingerAnnotations(annotations), ys: tf.oneHot([9], 10) };});

  const landmarksDataset = fist_landmarks_dataset.concatenate(one_landmarks_dataset).concatenate(two_landmarks_dataset)
    .concatenate(three_landmarks_dataset).concatenate(four_landmarks_dataset).concatenate(five_landmarks_dataset)
    .concatenate(six_landmarks_dataset).concatenate(seven_landmarks_dataset).concatenate(eight_landmarks_dataset).concatenate(nine_landmarks_dataset).batch(1);

  let handposesData = await landmarksDataset.toArray();
  // console.log(handposesData);

  let xs = [];
  let ys = [];
  handposesData.forEach((e) => {
    xs.push(e.xs.squeeze());
    ys.push(e.ys.squeeze());
  });

  const x = tf.stack(xs);
  const y = tf.stack(ys);
  // x.print();
  // y.print();

  const history = await model.fit(x, y, {
    batchSize: 1,
    epochs: 200, // Number of iterations over the entire dataset,
    shuffle: true
  });

  tfvis.show.history({name: 'History'}, history, ['loss', 'acc']);
}

trainModel();