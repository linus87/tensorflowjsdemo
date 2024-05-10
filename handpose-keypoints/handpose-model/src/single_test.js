const thumb = [50.298172  , 105.1985016, 143.2014923, 176.0051117];
const indexFinger = [197.6843719, 271.8681946, 327.4440918, 376.1433105];
const middleFinger = [192.1160889, 250.0362854, 227.1842194, 202.7913513];
const ringFinger = [182.1369324, 234.2925873, 199.1977539, 165.7783051];
const pinky = [170.9768982, 214.8193665, 195.7132874, 167.6916809];

function layerNormalization(x) {
  const epsilon = 1e-3;
  let {mean, variance} = tf.moments(x);
  return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, epsilon)));
}

layerNormalization(tf.tensor(thumb)).print();
layerNormalization(tf.tensor(indexFinger)).print();
layerNormalization(tf.tensor(middleFinger)).print();
layerNormalization(tf.tensor(ringFinger)).print();
layerNormalization(tf.tensor(pinky)).print();

const fingers = [thumb, indexFinger,middleFinger, ringFinger,pinky];

function batchNormalization(x, axis, gamma, beta) {
  const epsilon = 1e-3;
  let {mean, variance} = tf.moments(x, axis);
  return tf.add(tf.mul(tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, epsilon))), gamma), beta);
}

batchNormalization(tf.tensor(fingers), 0, 1, 0).print();