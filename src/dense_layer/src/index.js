// Tiny TFJS train / predict example.
let inputLayer, denseLayer;
async function run() {
    // Create a simple model.
    const model = tf.sequential();
    
    // Create a input layer node . Input shape is [none, 4]
    inputLayer = tf.layers.inputLayer({inputShape: [2, 4], batchSize: 2});
    model.add(inputLayer); 

    // kernelInitializer default value is glorot_uniform, biasInitializer default value is zeros.
    // Here, the kernel is ones, output shape is [1, 1]

    denseLayer = tf.layers.dense({units: 1, kernelInitializer:"ones", useBias: true});
    
    model.add(denseLayer);

    model.summary();

    /* Output is a tensor of shape [1, 4], it will render in the browser console as: 
     * Tensor
     * [[4],
     *  [4]]
     */
    model.predict(tf.ones([1, 2, 4])).print();
  }
  
  run();