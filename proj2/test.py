# Import pytorch package
import torch
from torch import FloatTensor
# Import self-defined package
import module as model
import loss as loss
import optimizer as optim
from util import one_hot_embedding, generate_disk_dataset, plot_with_labels
# Import math package
import math
import numpy as np
# Import Plot Package
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Model Training
def train_model(model, train_input, train_target, loss_fnc, nb_epochs, lr):
    '''
    Supervised machine learning model training.
    '''
    print("-------------------- Training --------------------")
    optimizer = optim.SGD(model.param(), lr)
    mini_batch_size = 100
    base = int(nb_epochs / 10)
    model.initParameters()
    l = []
    for e in range(nb_epochs + 1):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = loss_fnc(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if e % base == 0:
            print('Epochs', e, ': Loss ', loss.loss.item())
            l.append(loss.loss.item())
    print("--------------------------------------------------\n")
    return l
    
def compute_nb_errors(model, data_input, data_target):
    '''
    Compute number of mis-classified data
    '''
    nb_data_errors = 0
    data_output = model(data_input)
    
    _, target_classes    = torch.max(data_target.data, 1)
    _, predicted_classes = torch.max(data_output.data, 1)
    
    for k in range(data_input.size()[0]):
        if target_classes.data[k] != predicted_classes[k]:
            nb_data_errors = nb_data_errors + 1
    return nb_data_errors

# Mean square error
train_input, train_target = generate_disk_dataset(1000)
test_input, test_target = generate_disk_dataset(200)

Model_1 = model.MLP(model.Linear(2, 25), model.Tanh(),
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 2), model.sigmoid())

Model_2 = model.MLP(model.Linear(2, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 2), model.sigmoid())

Model_3 = model.MLP(model.Linear(2, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 2, initOption = 'Xavier'), model.sigmoid())

Model_4 = model.MLP(model.Linear(2, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 2), model.sigmoid())

learning_rate = 0.01
nb_epochs = 200
l_MSE = []
text = ['Tanh', 'ReLU + BN', 'ReLU + Init', 'ReLU']
for i, M in enumerate([Model_1, Model_2, Model_3, Model_4]):
    loss_fnc = loss.MSE(M)
    print('Model', i+1)
    print('Loss function: MSE; Activation function:', text[i])
    l_MSE.append(train_model(M, train_input, train_target, loss_fnc, nb_epochs, learning_rate))

    print("---------------------- Error ---------------------")
    nb_train_errors = compute_nb_errors(M, train_input, train_target)
    nb_test_errors = compute_nb_errors(M, test_input, 
                                       test_target)

    print('Test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    print('Train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))
    print("--------------------------------------------------\n")


# Cross-Entropy Loss
train_input, train_target = generate_disk_dataset(1000)
test_input, test_target = generate_disk_dataset(200)

Model_5 = model.MLP(model.Linear(2, 25), model.Tanh(),
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 25), model.Tanh(), 
                 model.Linear(25, 2))

Model_6 = model.MLP(model.Linear(2, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 25), model.ReLU(), model.BN(),
                 model.Linear(25, 2))

Model_7 = model.MLP(model.Linear(2, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 25, initOption = 'He'), model.ReLU(),
                 model.Linear(25, 2, initOption = 'Xavier'))

Model_8 = model.MLP(model.Linear(2, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 25), model.ReLU(),
                 model.Linear(25, 2))

learning_rate = 0.01
nb_epochs = 200
l_CE = []
text = ['Tanh', 'ReLU + BN', 'ReLU + Init', 'ReLU']
for i, M in enumerate([Model_5, Model_6, Model_7, Model_8]):
    loss_fnc = loss.CrossEntropy(M)
    print('Model', i+5)
    print('Loss function: Cross-Entropy; Activation function:', text[i])
    l_CE.append(train_model(M, train_input, train_target, loss_fnc, nb_epochs, learning_rate))

    print("---------------------- Error ---------------------")
    nb_train_errors = compute_nb_errors(M, train_input, train_target)
    nb_test_errors = compute_nb_errors(M, test_input, 
                                       test_target)

    print('Test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))
    print('Train error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                      nb_train_errors, train_input.size(0)))
    print("--------------------------------------------------\n")