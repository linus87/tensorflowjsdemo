```markdown {"id":"01J1VG6ET0G7WSE0EC1AH6X8BY"}

```

It was inconceivable, the argument numTokens of tf.layers.categoryEncoding will impact InputLayer's output shape. It means the last dimension of input must have the same size with numTokens.
In my opinion, it's a bug. And it's not fixed in current latest version 4.20.0.