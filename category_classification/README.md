```markdown {"id":"01J1VG6ET0G7WSE0EC1AH6X8BY"}

```

# Features of Embedding Layer
1. Dimensionality Reduction: The embedding layer maps high-dimensional input data into a lower-dimensional continuous space, reducing the number of parameters required.
2. Capturing Semantic Relationships: The embedding layer captures semantic relationships between inputs by representing similar inputs closer together in the embedding space.
3. Generalization: The embedding layer generalizes well to unseen inputs by learning meaningful representations from the training data.
4. Efficient Representation: The embedding layer provides an efficient representation of categorical or discrete data, such as words or categories.

# Demo
In this demo, it will create a model to judge if a number is even or odd. A number is even or odd, it's just one characteristic of number. So, the output size of embedding layer being one is enough. And you can see, after training, this model works.

And, this is also a binary-classification model demo.

# Pros and cons
## Pros
If you want to make decision on more characteristics, you just need to increase the output size of embedding layer.

## Cons
If more categories are given, the embedding layer need to be re-trained.