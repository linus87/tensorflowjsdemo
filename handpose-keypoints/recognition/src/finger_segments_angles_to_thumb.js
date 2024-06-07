// Tiny TFJS train / predict example.
// Define the model architecture
/**
 * 这个模型使用了两个特征：1）手指各个关节依序的角度；2）手指第一个节点与拇指第一个节点的角度（分辨1和8的区别）。
 * 模型准确度在96%左右。
 */
const model = tf.sequential();
model.add(tf.layers.inputLayer({inputShape: [5, 4], dtype: 'float32', batchSize: 1}));
// model.add(tf.layers.activation({activation: 'relu'}));
model.add(tf.layers.flatten());
const weights = tf.tensor( [[2.3594501  , -0.7945638 , -3.1529865 , -0.4711787 , 0.0567252  , -1.9171261, -0.608561  , -1.6480232, 1.0400684 , 1.933957  ],
  [2.5688214  , -1.2467136 , -8.6686611 , 0.1256213  , -15.6962042, 0.8564513 , 0.4758393  , 0.2260519 , 2.0739882 , 1.0225925 ],
  [4.2622824  , -0.0946206 , -0.3733386 , -5.7199826 , -4.1909456 , -0.7246298, 0.3767106  , 0.0597092 , 0.206615  , -0.582616 ],
  [3.2654185  , -0.6404571 , -2.6658778 , -0.9160498 , -3.7737005 , -1.9014739, 0.097749   , -0.749116 , 0.8354223 , 1.5851065 ],
  [-6.5261145 , 2.0462215  , 1.9004964  , -0.2506306 , 0.8263922  , -2.8208828, -1.3436809 , -3.6280792, -0.8313937, 2.7246072 ],
  [-19.8202019, 1.9356428  , -0.4625005 , -22.3309288, 0.5347077  , -1.9678738, -15.3524647, -1.2566333, 3.0410128 , -1.7032025],
  [0.3379785  , 1.4671724  , -0.9308149 , -0.96312   , -0.0618757 , -2.8157284, 0.496739   , -0.4622152, 1.8209684 , -2.2181838],
  [-0.8045448 , 2.6587598  , 1.0219752  , -1.9097695 , -0.0174198 , -2.1957929, -2.7537801 , 0.4449663 , -4.6283646, 2.627583  ],
  [-0.4272444 , -4.5701742 , 1.6511081  , 1.279054   , 0.7155008  , 1.2324463 , -3.3877022 , 3.2862532 , -0.0669133, 1.276675  ],
  [-6.5506949 , -21.1801491, 5.0035348  , 2.7694449  , 0.6801393  , 0.3163644 , -5.9091158 , 3.4532113 , -4.2245202, -3.0298941],
  [-1.6819971 , -1.215477  , 1.0168825  , 0.339157   , 0.3409925  , -0.6335835, 1.8250701  , 0.7607697 , 0.1874893 , -0.2175456],
  [0.1063126  , -0.9315354 , 1.5643344  , 1.3370059  , 0.361542   , -0.8933914, -3.7122934 , 5.1857233 , -5.1002979, 2.251188  ],
  [0.4901662  , -4.5341606 , 0.8352404  , 2.4531143  , 0.7640441  , 2.4123995 , -1.8900765 , -2.6440272, 1.9885197 , -0.9386849],
  [0.8854144  , 0.5032416  , -17.3394508, 3.2894094  , 2.4042189  , 3.7928617 , -6.0512848 , -3.8384001, -2.6262863, -4.2668839],
  [-0.3250794 , 0.3237696  , 0.2263854  , 0.3771758  , 0.58796    , 1.2131809 , 1.7171791  , 1.2351352 , -0.2423563, -2.8107219],
  [1.7879759  , -1.2490838 , -1.5901256 , 1.1818938  , 1.1811754  , -0.4341877, -3.2903068 , 2.6618497 , -1.4583679, 0.8511657 ],
  [-7.9927592 , 1.6668451  , -0.1252998 , 2.2209301  , 1.7771438  , 4.043704  , 6.7545514  , -5.5072184, -2.1400151, -2.2671378],
  [-2.3071339 , 1.1652792  , 4.8272214  , 1.6526816  , 2.0325873  , 3.3348181 , 6.0619936  , -3.3909957, -4.3693681, -2.1624765],
  [-0.5908294 , -0.5979117 , -0.9994282 , 0.9272138  , 0.6839678  , 1.0973731 , 3.2276175  , -1.318221 , 0.8200408 , -2.6404133],
  [0.5112475  , 0.5588636  , -0.6475795 , -1.1019403 , 0.922231   , -2.1403666, 2.7919035  , 1.2903314 , 0.4588671 , -3.3804324]]
);
const bias = tf.tensor( [3.4945703, 0.2323064, -1.1229998, -0.5829316, 0.141202, -3.0699074, -0.5127799, -1.3024848, -0.1441878, 1.8308026]
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

function convertVectorsIntoAnglesAll(annotations) {
  const fiveFingersTensors = convertAnnotationsIntoVector(annotations);
  // fingerTensors.print();
  const fiveFingesrSegmentLengthTensors = tf.norm(fiveFingersTensors, 2, 2, true);
  const fiveFingerSegmentLengthArray = fiveFingesrSegmentLengthTensors.arraySync();
  // fingerSegmentLengthTensors.print();

  const fiveFingerVectors = fiveFingersTensors.arraySync();
  const base = fiveFingerVectors[0][2];
  let angles = [];
  for (let i=0; i<5; i++) {
    const segment = fiveFingerVectors[i][1];
    const dot = base[0] * segment[0] + base[1] * segment[1] + base[2] * segment[2];
    angles.push(dot / fiveFingerSegmentLengthArray[0][1] / fiveFingerSegmentLengthArray[i][1]);
  }
  
  return tf.tensor(angles);
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
  // angles.print();
  const distances = convertVectorsIntoAnglesAll(annotations).reshape([5, 1]);
  // distances.print();
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