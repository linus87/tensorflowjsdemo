/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';

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

function convertVectorsIntoAnglesWithFirstSegment(annotations) {
  const fingerTensors = convertAnnotationsIntoVector(annotations);
  fingerTensors.print();
  const fingerSegmentLengthTensors = tf.norm(fingerTensors, 2, 2, true);
  fingerSegmentLengthTensors.print();

  const fingerVectors = fingerTensors.arraySync();
  const fingerSegmentVectorsDot = fingerVectors.map(segments => segments.reduce((accumulator, currentValue, currentIndex, array) => {
    let previous = array[0];
    if (currentIndex === 1) return [currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]];
    else return accumulator.concat(currentValue[0] * previous[0] + currentValue[1] * previous[1] + currentValue[2] * previous[2]);
  }));

  const fingerSegmentLengths = fingerSegmentLengthTensors.arraySync();
  const fingerSegmentsAngles = fingerSegmentVectorsDot.map((fingerSegments, fingerIndex) => fingerSegments.map((segment, segmentIndex) => segment / fingerSegmentLengths[fingerIndex][0] / fingerSegmentLengths[fingerIndex][segmentIndex + 1]) );
  console.log(fingerSegmentsAngles);
  
  return tf.tensor(fingerSegmentsAngles);
}

function convertVectorsIntoAnglesToThumb(annotations) {
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
    angles.push(dot / fiveFingerSegmentLengthArray[0][2] / fiveFingerSegmentLengthArray[i][1]);
  }
  
  return tf.tensor(angles);
}

function mergeFingerAnnotations(annotations) {
  const angles = convertVectorsIntoAnglesWithFirstSegment(annotations);
  // angles.print();
  const distances = convertVectorsIntoAnglesToThumb(annotations).reshape([5, 1]);
  // distances.print();
  angles.concat(distances, 1).print();
  return angles.concat(distances, 1);
}

const classNames = ['fist', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
console.log(tf);
const numericHanposeModel = tf.sequential();
numericHanposeModel.add(tf.layers.inputLayer({inputShape: [5, 4], dtype: 'float32', batchSize: 1}));
// numericHanposeModel.add(tf.layers.activation({activation: 'relu'}));
numericHanposeModel.add(tf.layers.flatten());
const weights = tf.tensor([[0.0985235 , -0.6725039, -1.2263875, -0.0937763, 0.5963337 , -1.0649683, -0.3961747, -0.2379668, 0.2905979 , 0.0608547 ],
  [-1.3550634, -0.2655806, -3.1202619, 1.3800374 , -7.3700428, 0.9856706 , 0.2600968 , -1.0554063, 1.5129522 , -0.1480823],
  [-1.5781322, -2.2796509, -4.3344288, -2.42031  , -5.0786481, 2.7798607 , 1.4855461 , 0.8891832 , 3.1994624 , -1.5210412],
  [-0.0366645, -0.6371205, -0.9775266, -0.0926177, -3.3165455, 0.0347334 , -0.0983607, 0.9136122 , 0.9690129 , -0.1066127],
  [-1.4887283, 0.3831153 , 0.433804  , 0.1670349 , 0.0479938 , -1.0575091, -0.0960088, -1.5847441, -0.4718798, 1.0536159 ],
  [-9.7079906, 2.0049376 , 0.3559208 , -4.9729481, 0.9168058 , 0.2879395 , -3.8547606, -2.1058066, 0.4426471 , 0.4375353 ],
  [-5.7502975, 2.7114549 , 0.7774933 , -4.6634321, 0.8895071 , 0.6071057 , -3.3791742, -1.0654057, 1.1403334 , -3.4309611],
  [-0.3887093, 1.3558276 , -1.9523437, 0.2702711 , -1.787586 , 0.6456208 , -0.5642493, 1.9637153 , -3.2067657, 1.1430703 ],
  [0.2698068 , -0.7911412, 0.0506371 , 0.0793767 , 0.5682659 , 0.094432  , -0.2798071, 0.9040821 , -0.3197476, 0.7527086 ],
  [-2.0634866, -1.7396237, 2.7190077 , 1.5125903 , 0.7885121 , 1.0997355 , -0.8157103, 3.424114  , 0.6641881 , -1.0728474],
  [-0.7934334, -0.434527 , 3.1834834 , 1.4401653 , 0.7600469 , 1.4012491 , -1.2341585, 0.8431059 , 0.1802796 , -1.382588 ],
  [-1.1354349, -0.1803125, 0.0434342 , 0.3066014 , -0.838285 , 0.0248158 , -1.4463301, 2.7075381 , -2.9135816, 1.4200735 ],
  [1.4180015 , -1.1864495, 0.7817494 , 0.3491427 , 0.4054699 , 0.5627315 , 0.3232037 , -1.6530511, 0.9955652 , 0.561291  ],
  [-0.0981907, -0.1780868, -3.3392837, 2.2062073 , 1.4402421 , 2.1553123 , -0.4165385, 0.2327493 , -0.0115665, -0.2134603],
  [-1.1533828, 0.2030584 , -2.2000394, 1.2616612 , 1.2276801 , 2.7671597 , -1.4319299, -1.6073364, -0.401849 , 0.3427812 ],
  [-0.29368  , -0.6983641, 0.4207021 , -0.1197562, -0.4247456, 0.1004268 , -0.9685546, 2.0843844 , -1.7460412, 0.5301157 ],
  [-2.5319591, 1.5839123 , 0.8829629 , 1.0084035 , 0.6661327 , -0.1621108, 1.8903419 , -4.2818165, 0.2969776 , -0.9362968],
  [-1.3081889, 0.0037497 , 3.6093566 , 1.4722894 , 1.2488894 , 2.1109562 , 4.5408335 , -0.259241 , -0.629019 , -0.6904672],
  [-1.0627041, 0.5833057 , -1.8352082, 1.3834679 , 1.6975383 , 2.5533414 , 3.2776897 , -0.1557869, -1.6216304, -0.2878756],
  [-0.7649097, 0.2990404 , -0.0136587, -0.9813275, 0.4241155 , -0.5998385, -0.5051569, 0.8175391 , 0.569575  , -2.3579574]]
);
const bias = tf.tensor( [0.0538871, -0.2203505, -0.5859668, -0.5580947, 0.3606549, -0.9791526, -0.4906588, 0.2716785, 0.1118058, 0.1563296]);
numericHanposeModel.add(tf.layers.dense({weights: [weights, bias], units: 10, activation: 'softmax'}));

// Compile the model with a binary loss function and an optimizer
numericHanposeModel.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

/****** Segment line ******/

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}
tfjsWasm.setWasmPaths({
  'tfjs-backend-wasm.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`,
  'tfjs-backend-wasm-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-simd.wasm`,
  'tfjs-backend-wasm-threaded-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-threaded-simd.wasm`,
});
let videoWidth, videoHeight, rafID, ctx, canvas, ANCHOR_POINTS,
  scatterGLHasInitialized = false, scatterGL, fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
  };  // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;

const state = {
  backend: 'webgl',
  renderPointcloud: renderPointcloud
};

function drawPoint(y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawKeypoints(keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(x - 2, y - 2, 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(points, false);
  }
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

let video;
async function main() {
  await tf.setBackend(state.backend);
  if (!tf.env().getAsync('WASM_HAS_SIMD_SUPPORT') && state.backend == "wasm") {
    console.warn("The backend is set to WebAssembly and SIMD support is turned off.\nThis could bottleneck your performance greatly, thus to prevent this enable SIMD Support in chrome://flags");
  }
  model = await handpose.load();

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = 'red';
  ctx.fillStyle = 'red';

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  // ANCHOR_POINTS = [
  //   [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
  //   [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  // ];

  landmarksRealTime(video);
}

const handposeImg = document.getElementById('current-pose');

const landmarksRealTime = async (video) => {
  async function frameLandmarks() {

    ctx.drawImage(
      video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
      canvas.height);
    const predictions = await model.estimateHands(video);
    // console.log(predictions);
    if (predictions.length > 0) {

      const handposePredicts = numericHanposeModel.predict(mergeFingerAnnotations(predictions[0].annotations).reshape([1, 5, 4]));

      handposePredicts.argMax(1).data().then(index => {
          console.log(`${classNames[index]}`);
          handposeImg.src = `poses/${classNames[index]}.png`;
      });

      const result = predictions[0].landmarks;
      drawKeypoints(result, predictions[0].annotations);

    }
    rafID = requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

main();

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;