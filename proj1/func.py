# Import Pytorch Package 
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# Import Tool package
import dlc_practical_prologue as prologue
# Import Plot Package
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Other Package
import warnings
warnings.filterwarnings('ignore')

mini_batch_size = 50
N = 1000

def trainBaselineNet(model, train_input, train_target, nb_epochs = 25, lambda_l2 = 0.1, mini_batch_size = 50, lr = 1e-3*0.5): 
    """
    Train BaselineNet model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3*0.5)
    
    loss_list = []
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            # Forward pass
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            
            # Add L2-regularazation
            for p in model.parameters():
                loss += lambda_l2 * p.pow(2).sum()
            
            # Back-propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())
    return loss_list


def errorsCompute(model, data_input, data_target):
    """
    Compute error/accurancy of model.
    """
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        # Prediction
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(F.softmax(output.data, dim = 0), 1)
        
        # Compare and count error
        for k in range(int(mini_batch_size)):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def PipelineBase(Net, rounds = 15, mini_batch_size = 50, N = 1000):
    """
    Use "mini_batch_size" to optimize step by step in "rounds" rounds.
    """
    loss_round_list = []
    train_errors_list, test_errors_list = [], []
    
    for k in range(rounds):
        # Generate Data
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)

        # Model training
        model = Net()
        loss_list = trainBaselineNet(model, train_input, train_target)
        loss_round_list.append(loss_list)

        # Predict and compute error
        nb_train_errors = errorsCompute(model, train_input, train_target)
        nb_test_errors = errorsCompute(model, test_input, test_target)
        train_errors_list.append(nb_train_errors/train_input.size(0))
        test_errors_list.append(nb_test_errors/test_input.size(0))

        # Logging
        print('Iteration ', k+1)
        print("---------------------- Error ---------------------")
        print('Test error: {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
        print('Train error: {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                          nb_train_errors, train_input.size(0)))
        print("--------------------------------------------------\n")
    return model, loss_round_list, train_errors_list, test_errors_list


def estiPerform(train_errors_list, test_errors_list, rounds=15):
    trainErrorAvg = torch.FloatTensor(train_errors_list).mean().item()
    trainErrorStd = torch.FloatTensor(train_errors_list).std().item()
    testErrorAvg = torch.FloatTensor(test_errors_list).mean().item()
    testErrorStd = torch.FloatTensor(test_errors_list).std().item()
    print("Estimates of {} rounds:".format(rounds))
    print("Train error Average: {:3f}; Train error standard deviations: {:3f}".format(trainErrorAvg, trainErrorStd))
    print("Test error Average: {:3f};  Test error standard deviations: {:3f}".format(testErrorAvg, testErrorStd))

    plt.figure(figsize=(10, 5))
    ax2 = plt.subplot(111)
    ax2.plot(range(len(train_errors_list)), train_errors_list, label='Train Error', marker='o')
    ax2.plot(range(len(test_errors_list)), test_errors_list, label='Test Error', marker='o')
    ax2.set_ylabel('Error rate')
    ax2.set_xlabel('round')
    ax2.legend()
    ax2.set_title('Train/Test error rate standard deviations')
    plt.show()
    
    
def trainCNNAuxiliaryLoss(model, train_input, train_class, train_target, nb_epochs = 25, lambda_l2 = 0.1, mini_batch_size = 50, lr = 1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    loss_list = []
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            # Forward pass
            output_multi, output_single = model(train_input.narrow(0, b, mini_batch_size))
            # Main loss + Auxiliary loss
            loss =  criterion(output_multi, train_class.narrow(0, b, mini_batch_size)) + \
                    criterion(output_single, train_target.narrow(0, int(b/2), int(mini_batch_size/2)))
            
            # Add L2-regularazation
            for p in model.parameters():
                loss += lambda_l2 * p.pow(2).sum()
            
            # Back-propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.item())
            
    return loss_list


def errorsComputeAux(model, data_input, data_target, mini_batch_size = 50):
    """
    Compute error/accurancy of model with auxiliary loss.
    """
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        # Prediction
        output_multi, output_single = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(F.softmax(output_single.data), 1)
        
        # Compare and count error
        for k in range(int(mini_batch_size/2)):
            if data_target.data[int(b/2) + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors


def PipelineAux(Net, rounds = 15, mini_batch_size = 50, N = 1000):
    """
    Use "mini_batch_size" to optimize step by step in "rounds" rounds.
    """
    loss_round_list = []
    train_errors_list, test_errors_list = [], []
    
    for k in range(rounds):
        # Generate Data
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
        
        # Split the pairs of images in data
        train_input_pic = train_input.view(-1, 1, 14, 14)
        train_classes_pic = train_classes.view(-1)
        test_input_pic = test_input.view(-1, 1, 14, 14)
        test_classes_pic = test_classes.view(-1)
        
        # Model training
        model = Net()
        loss_list = trainCNNAuxiliaryLoss(model, train_input_pic, train_classes_pic, train_target)
        loss_round_list.append(loss_list)

        # Predict and compute error
        nb_train_errors = errorsComputeAux(model, train_input_pic, train_target)
        nb_test_errors = errorsComputeAux(model, test_input_pic, test_target)
        train_errors_list.append(nb_train_errors/train_input.size(0))
        test_errors_list.append(nb_test_errors/test_input.size(0))
        
        # Logging
        print('Iteration ', k+1)
        print("---------------------- Error ---------------------")
        print('Test error: {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                          nb_test_errors, test_input.size(0)))
        print('Train error: {:0.2f}% {:d}/{:d}'.format((100 * nb_train_errors) / train_input.size(0),
                                                          nb_train_errors, train_input.size(0)))
        print("--------------------------------------------------\n")
    return model, loss_round_list, train_errors_list, test_errors_list
