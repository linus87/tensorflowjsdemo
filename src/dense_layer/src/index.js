// Tiny TFJS train / predict example.
async function run() {
    // Create a simple model.
    const model = tf.sequential();
    
    // Calling layers.inputLayer() using add() method 
    model.add(tf.layers.inputLayer({inputShape: [4]})); 

    // kernelInitializer default value is glorot_uniform, biasInitializer default value is zeros.
    model.add(tf.layers.dense({units: 1, kernelInitializer:"ones", useBias: true}));

    /* Output is a tensor of shape [1, 4], it will render in the browser console as: 
     * Tensor
     * [[4],]
     */
    model.predict(tf.ones([1, 4]), {batchSize:1}).print();

  }
  
  run();