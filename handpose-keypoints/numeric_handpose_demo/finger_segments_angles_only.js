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

const classNames = ['fist', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
console.log(tf);
const numericHanposeModel = tf.sequential();
numericHanposeModel.add(tf.layers.inputLayer({inputShape: [5, 3], dtype: 'float32', batchSize: 1}));
numericHanposeModel.add(tf.layers.activation({activation: 'relu'}));
numericHanposeModel.add(tf.layers.flatten());
const weights = tf.tensor([[5.3359742  , 0.2504182  , -3.2755277 , -0.356792 , 0.2157451 , -5.3149023, -0.889978 , -1.5613234, -0.2194791 , 1.3663721 ],
  [3.2457666  , -5.040802  , -15.6675749, 0.4846437 , -20.693924, 5.4521108 , 0.279647  , 0.4954756 , 4.8911152  , 2.5759835 ],
  [3.6197002  , 0.1140838  , -0.7106852 , -6.4531431, -7.0236239, 0.5817383 , 0.2993872 , -0.3905171, -0.7119688 , 1.1290802 ],
  [-8.7171745 , 3.6741428  , 2.1097219  , 0.5361198 , 1.7913443 , -2.0627499, -1.9234776, -4.2802649, -4.0456228 , 3.1457255 ],
  [-20.3464146, 3.5133173  , 0.8957437  , -20.758585, 0.9730878 , -1.8194798, -14.335393, 0.377414  , -0.7355188 , -1.2753879],
  [-1.0204698 , 2.4139938  , 0.0146416  , -0.5868888, -0.0632308, -5.2803278, -0.0384194, 0.0596275 , -1.7681683 , 0.5182613 ],
  [-0.3335978 , -1.827581  , 1.9562269  , 1.3186108 , 0.7485022 , -0.6725502, -7.3675351, 3.2003372 , -2.0012162 , 1.8172157 ],
  [-6.5675364 , -17.8478794, 15.0993185 , 5.2419372 , 1.5628178 , 1.394116  , -4.9067311, 14.0417938, -19.6844501, 1.6747411 ],
  [1.3478674  , -1.6898286 , 1.1083258  , 0.2772829 , 0.7841425 , -3.3632491, 1.9247382 , 4.4572763 , 1.1817255  , -2.3176966],
  [-1.1688169 , -3.6474802 , 2.8227537  , 2.0138359 , 1.4008156 , 0.7989523 , -5.1820087, 1.5045717 , 0.9474033  , 1.8527193 ],
  [1.9650421  , -5.6037149 , -23.5310001, 5.700376  , 3.6753755 , 7.5065069 , -4.124588 , -1.7082171, -9.2514238 , -3.2355821],
  [2.301966   , 0.7771633  , 1.6481497  , 0.4527129 , 0.5620683 , 1.6487997 , 2.0477746 , -1.0373251, 1.0322257  , -3.2615077],
  [-6.4934449 , -0.1113763 , 0.9194624  , 1.2276284 , 2.7076108 , 2.0047572 , 5.9351768 , -8.4957151, 2.22001    , -9.00348  ],
  [-5.756978  , -0.5788895 , 5.818768   , 2.8158348 , 2.5551639 , 5.8851004 , 9.9776859 , -7.5211015, -15.2506571, -1.4337714],
  [2.1238959  , -0.9585925 , -0.9083371 , 0.0458024 , 0.8583028 , 1.4668708 , 2.6700459 , -0.7493712, 1.8534898  , -2.8392944]]
);
const bias = tf.tensor([5.972281, 1.4995559, -0.6347349, -0.7851848, 0.6185119, -5.9113107, -1.4548841, -1.5542352, -1.6470259, 0.7666029]
);
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

      const handposePredicts = numericHanposeModel.predict(convertVectorsIntoAngles(predictions[0].annotations).reshape([1, 5, 3]));

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