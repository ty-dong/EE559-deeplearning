# Import Pytorch Package 
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
# Import Tool package
import dlc_practical_prologue as prologue
# Import the models and functions
from func import *
from models import *

# Load Data
N = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)

# Set training parameters
inpDim = train_input.view(N, -1).size()[1] # inpDim: Input dimension
nb_hidden1, nb_hidden2 = 160, 128 # nb_hidden1, nb_hidden2: Hidden dimensions
rounds = 15
mini_batch_size = 50

# Baseline Model
print("Train Baseline Model:")
print("--------------------------------------------------")
model1, loss_round_list1, train_errors_list1, test_errors_list1 = PipelineBase(BaselineNet, rounds, mini_batch_size, N)
estiPerform(train_errors_list1, test_errors_list1)
print("The parameters number of Baseline Model:", sum(p.numel() for p in model1.parameters()),'\n')

# CNN Model
print("Train CNN Model:")
print("--------------------------------------------------")
model2, loss_round_list2, train_errors_list2, test_errors_list2= PipelineBase(CNN, rounds, mini_batch_size, N)
estiPerform(train_errors_list2, test_errors_list2)
print("The parameters number of Baseline Model:", sum(p.numel() for p in model2.parameters()), '\n')

# CNN Model with Auxiliary Loss
print("CNN Model with Auxiliary Loss:")
print("--------------------------------------------------")
model3, loss_round_list3, train_errors_list3, test_errors_list3 = PipelineAux(CNNAuxiliaryLoss, rounds, mini_batch_size, N)
estiPerform(train_errors_list3, test_errors_list3)
print("The parameters number of Baseline Model:", sum(p.numel() for p in model3.parameters()), '\n')


