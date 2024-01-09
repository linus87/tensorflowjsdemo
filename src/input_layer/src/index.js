// Tiny TFJS train / predict example.
async function run() {
    // Create a simple model.
    const model = tf.sequential();
    
    // Calling layers.inputLayer() using add() method 
    model.add(tf.layers.inputLayer({inputShape: [4]})); 

    /* Output is a tensor of shape [1, 4], it will render in the browser console as: 
     * Tensor
     * [[1, 1, 1, 1],]
     */
    model.predict(tf.ones([1, 4])).print();

  }
  
  run();