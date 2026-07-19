My implementation of Makemore, following the Andrej Karpathy series "Zero to Hero" (https://karpathy.ai/zero-to-hero.html)

## Part 1: Statistical Modeling and Single Layer NN
This part contains two models that learn from the same probability distribution; they learn to predict the next character given the current character (a *bigram* model).

For both models, a negative log likelihood function is used to calculate the data loss. This is used due to the classification nature of the models: we classify the next character by assigning a probability to a discrete number of characters to choose from. Critically, finding the likelihood allows us to assign a loss depending on how confident the model's probability was in the correct answer. The log, negative, and normalization portions are for making the data cleaner and easier to work with.

The first model is a simple statistical model that calculates the raw probability between every combination of letters. It does this by creating a matrix with every possible bigram (a single '.' to encode start/end and the 26 letters of the alphabet, so 27x27) and counts up every time they occur in the training dataset. Then, it normalizes the rows such that every letter can be mapped to a probability distribution of letters that may follow it. Finally, the next character to choose is sampled from this probability distribution.

The second model is a single layer of 27 neurons (one logit for each possible next character). The key difference between this model and the prior, is that the prior model is in a fixed state. We can calculate raw probabilities and find the subsequent NLL, but there are no parameters by which we can attempt to optimize (lower) that loss. In this scenario where we only have one layer, and given that since we trying to represent probabilities via passing logits through softmax, we end up at the same lower bound as the statistical model. Fundamentally this is because the neurons effectively end up with the same direct representation as the statistical model: A 1x27 tensor lookup table of probabilities.
- This model also includes the idea of renormalization loss. This acts as a parameter for model tuning by adding some avg. weights^2 term such that weights closer to zero incur less loss. This is favorable for ensuring that weights do not swing too far in the direction of a small data sample too quickly, and ensuring that model weights stay lower.

## Part Two: Multi-Layer Perceptron NN
This part transitions from the basic, identical statistical/single-layer models of the last part to implementing prediction using a MLP.

There are a few key properties that grants the MLP an advantage over the statistical/single-layer models.

First, the MLP introduces a "hidden layer" which acts as a layer of neurons in between the input layer and the output layer. The key distinction here is that the hidden layer applies the `tanh` function to its output. `tanh` introduces an element of non-linearity into the model that prevents the model from collapsing to learning a linear transformation, and as such learn much more complex patterns than the linear of a single-layer model.

Second, the MLP samples from `block_size` prior characters as its input rather than just the prior character. In addition, it utilizes an embedding space that each character is placed into, which through training helps the model learn the relationships and similarities between different characters. Notably these are not pre-programmed embeddings, but parameters that are trained like the rest of the network.

As an additional note, this part also introduced the idea of splitting data into training/validation/test sets. Splitting the data like this is critical to preventing the model from overfit on the training data and fails to generalize to new scenarios; as with regularization loss it helps to combat overfitting.
- Generally, the parameters (embeddings, weights, biases, etc.) of a network should be trained on the training set, whereas the hyperparameters are adjusted based on the validation set. Finally the test set is used to evaluate the models performance.
