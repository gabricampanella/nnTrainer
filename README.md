# resnetTrainer
Implementation of a ResNet for learning on new datasets. Adapted from Facebook's implementation.

## Usage details:
th main.lua -h -- brings up the help with all options
input (-data option) is a .t7 file formatted as follows: table of 3 elemets
1) train: table of 2 elements (data: 4D tensor, labels: 1D tensor)
2) val: table of 2 elements (data: 4D tensor, labels: 1D tensor)
3) test: table of 2 elements (data: 4D tensor, labels: 1D tensor)
When data is loaded batches are formed by first randomizing the order of input.
Weights of the model are updated at every mini-batch. At end of each epoch the overall loss and accuracy error (for classification) are calculated and saved to a .csv file. Models at the end of each epoch and and the config file for the optimizer are also saved. At every end of epoch the current model is compared to best model (according to loss, not accuracy) and in case of improvement, best model is overwritten.
In the end the test set is used to predict on the best model and the predictions are saved to a .csv file
