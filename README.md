# Advanced Fuzzy Relational Neural Network (AFRNN)

Implementation of [paper](https://ceur-ws.org/Vol-3074/paper27.pdf) presented at WILF2021

## Important
The code in the repository is an update of the paper linked above. It contains some improvements and results 
that are better than those reported in the article. It is currently under development and research 
and any advances will be reported here

### Differences

* The code is rewritten in [PyTorch](https://pytorch.org/)
* [Captum](https://captum.ai/) library is used for exaplinability. Some layer can be slightly different and improved.
* Constraints over gradients are changed from MinMaxScaling to Clipping. This modification highly helps the network to learn and remove the heavy oscillation of the published results over the loss function.
* There are more layers such as Łukasiewicza/Yager ops.
* It is very **experimental**, it should be more **pythonic**

### Differences on results

![old_results.png](images%2Fold_results.png)

*Table 1. Old results published in WILF2021*

<br />

![new_results.png](images%2Fnew_results.png)

*Table 2. New results, red cells represents tests not done on MNIST because it is a too easy dataset*

## Built-in Fuzzy Layers

* MaxMin2d
* MaxYager2d
* LeakyThreshold


## Built-in Models

* Conv2d
* MaxMinAFRNN
* MaxMin2LAFRNN
* MaxLukAFRNN
* MaxLuk2LAFRNN
* MaxLuk3LAFRNN
* MaxLukMaxMinAFRNN
* MaxLearnableYagerAFRNN
* LeNet5AFRNN
* LeNet5SigAFRNN
* LeNet5_2AFRNN
* LeNet5_2SigAFRNN
* LeNet5
* LeNetLukAFRNN


# Execution
For training, testing and explainability it is possible to use the following parameters:

| **Program_Parameters** | **Default** | **Usage**                                                                                                                              |
|------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------|
| --name                 | ''          | Name the experiment                                                                                                                    |
| --weights              | ''          | Path to weights to load the pretrained network                                                                                         |
| --resume_epoch         | -1          | Epoch from which resume training                                                                                                       |
| --dataset              | mnist       | Dataset to use. It uses torchvision.datasets                                                                                           |
| --mode                 | train       | Specifies the action to perform. Can be [train - test - explain - attack]                                                              |
| --path                 | ''          | Path where save the model and tests                                                                                                    |
| --batch_size           | 8           | Mini batch size                                                                                                                        |
| --model                | fuzzy       | Name of the model to run. All models are loaded from models.py. Insert new models there. Please use AFRNN as placeholder in new models |
| --epochs               | 1000        | Number of epochs                                                                                                                       |
| --tnorm_p              | 1.          | *p* of the t-norm (used in Yager)                                                                                                      |
| --num_classes          | 10          | Number of classes                                                                                                                      |
| --constraint           | None        | Constraint type. Can be [clip - minmax - sigmoid - gaussian - sinc2]                                                                   |
| --log_every            | 10000       | Number of  iteration before log and save to file using tensorboard                                                                     |

## Cite us
```
@inproceedings{di2021advanced,
  title={Advanced Fuzzy Relational Neural Network.},
  author={Di Nardo, Emanuel and Ciaramella, Angelo},
  booktitle={WILF},
  year={2021}
}
```
