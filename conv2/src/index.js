// Tiny TFJS train / predict example.

let fromBlobImgElement = document.getElementById('fromBlobImg');

async function loadImage(imageUrl) {
  return fetch(imageUrl)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
    })
    .then(blob => {
      return blob;
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

// Tiny TFJS train / predict example.
let inputLayer, denseLayer;
async function run() {
  const imageBlob = await loadImage('./cat.jpeg');
  console.log(imageBlob) ;
  fromBlobImgElement.src = URL.createObjectURL(imageBlob);

  const imageSize = 100;
  const kernelSize = 2;

  let {blob, rgbTensor} = await compressImage(imageBlob, imageSize, imageSize);
    
  const img = document.getElementById('toBlobImg');
  img.src = URL.createObjectURL(blob);

  console.log(rgbTensor);

  // Define the shape of the filter
  filter_shape = [3, 3, 3, 1];

  const filter = tf.variable(tf.randomNormal(filter_shape));
  filter.print();

  const firstConv = tf.conv2d(rgbTensor, filter, [1, 1], 'same').toInt();
  const values = await firstConv.array();

  // Render to visor
  const surface = { name: 'Heatmap', tab: 'Charts' };
  tfvis.render.heatmap(surface, {values});
  

  // // Create a simple model.
  // const model = tf.sequential();
  
  // // Create a input layer node . Input shape is [none, 4]
  // inputLayer = tf.layers.inputLayer({inputShape: [imageSize, imageSize, 3], batchSize: 1});
  // model.add(inputLayer); 


  // const conv2d = tf.layers.conv2d({filters:1, kernelSize:kernelSize, strides:2, padding:'same'});
  // model.add(conv2d);

  // model.summary();

  // // Compile the model
  // model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

  

  // // kernelInitializer default value is glorot_uniform, biasInitializer default value is zeros.
  // // Here, the kernel is ones, output shape is [1, 1]

  // denseLayer = tf.layers.dense({units: 1, kernelInitializer:"ones", useBias: true});
  
  // model.add(denseLayer);

  

  /* Output is a tensor of shape [1, 4], it will render in the browser console as: 
    * Tensor
    * [[4],
    *  [4]]
    */
  // model.predict(tf.ones([1, 2, 4])).print();
}
  
run();