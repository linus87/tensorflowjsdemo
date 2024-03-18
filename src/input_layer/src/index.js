// Tiny TFJS train / predict example.
async function run() {
    // Create a simple model.
    const model = tf.sequential();
    
    // create a input layer node, input shape is [none, 4]
    model.add(tf.layers.inputLayer({inputShape: [4], batchSize: 2}));

    model.summary();

    /* Output is a tensor of shape [2, 4], it will render in the browser console as: 
     * Tensor
     * [[1, 1, 1, 1],
     *  [1, 1, 1, 1],
     *  [1, 1, 1, 1]]
     */
    model.predict(tf.ones([3, 4]), {batchSize:2}).print();

  }
  
  run();