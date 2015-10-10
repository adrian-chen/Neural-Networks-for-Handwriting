# Training Deep Neural Networks
## Author: Adrian Chen

Before attempting to run this code, [theano](http://deeplearning.net/software/theano/) must be installed

The code can be run using:
```
$ python train.py
```

## In this program, I utilize several established tools:
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

## Addition Unimplemented Improvements

Larger datasets
* Meaningful improvements to the accuracy of a model can be obtained with larger training data sets
* Data can be artficially obtained by rotating, skewing, or translating images

## Stacked DNNs
This is an idea I had while researching DNNs. A highly complex Neural Network might be best implemented in multiple steps. This is because in networks that aren't shallow, we might run into the issues of exploding and vanishing gradients, where the speed of learning at the end or beginning will be severely inhibited.
* For example in face recognition, train a network to idenfity eye color, hair color, and other important heuristics. Then after this network is trained, keep the values static and use it as the inputs to a new network which might decide race, identity, etc.
### Problems