# Training Deep Neural Networks
## Author: Adrian Chen

Before attempting to run this code, [theano](http://deeplearning.net/software/theano/) must be installed.

This code attempts to accurately train a model to recognize data from the MNIST handwriting dataset.

My accuracy on the validation set is currently 98.99% and my accuracy on the test set is 98.91%

The code can be run using:
```
$ python train.py
```

Within ```train.py``` the neural network can be altered in several ways.

```
mini_batch_size = 10
net = Network([
    ConvPoolLayer(
    	image_shape=(mini_batch_size, 1, 28, 28), 
	    filter_shape=(20, 1, 5, 5), 
	    poolsize=(2, 2)),
    ConvPoolLayer(
    	image_shape=(mini_batch_size, 20, 12, 12), 
        filter_shape=(40, 20, 5, 5), 
        poolsize=(2, 2)),
    FullyConnectedLayer(n_in=40*4*4, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
    validation_data, test_data)
```

## Tools for higher accuracy:
Convolutional Neural Networks
* Group together connected inputs. Ex: pixels near each other in a picture.

Backpropagation
* Faster computation
* Compute gradient of the cost function

Cross-Entropy Cost Funtion
* Traditional cost function train slowly when the output is very different from the correct value
* Allows fast drastic error correction in the output.

Softmax Layer
* Allows simple estimation of the estimated probability that an output is correct
* Total of all probabilities is 1

Regularization
* Weight decay to reduce the size of weights while still minimizing the cost function
* Reduce Overfitting

Dropout
* Train the model by disabling different sets of hidden neurons while training the model
* Reduced reliance on the outputs of nearby neurons
* Forced to model more robust features

## Additional Unimplemented Improvements

Larger datasets
* Meaningful improvements to the accuracy of a model can be obtained with larger training data sets
* Data can be artficially obtained by rotating, skewing, or translating images

## Stacked DNNs
This is an idea I had while researching DNNs. A highly complex Neural Network might be best implemented in multiple steps. This is because in networks that aren't shallow, we might run into the issues of exploding and vanishing gradients, where the speed of learning at the end or beginning will be severely inhibited.
* For example in face recognition, train a network to identify eye color, hair color, and other important heuristics. Then after this network is trained, keep the values static and use it as the inputs to a new network which might decide race, identity, etc.
## Problems
* This requires a substantially larger dataset, which may not be feasible to gather.
* Identifiable sub-characteristics might be harder for a human to identify than a computer in some circumstances.

