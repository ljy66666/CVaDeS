import torch
import numpy as np
from DataLoader import load_BRCA
from train import trainnet
from km_logrank import km_logrank_picture


device_id = 0
torch.cuda.set_device(device_id)
dtype = torch.FloatTensor

''' Net Settings'''
feature_size = 250 
class_size = 1 
latent_size = 10 

''' Initialize '''
Initial_Learning_Rate = [0.03, 0.01, 0.001, 0.00075]
L2_Lambda = [0.1, 0.01, 0.005, 0.001]
num_epochs =30  ###for grid search
Num_EPOCHS =50  ###for training

''' load data  
need to provide the absolute file path.
'''
name='BRCA'
x_train, ytime_train, yevent_train = load_BRCA(f"/Data/ljy/single_cox250/{name}_train_250.csv", dtype)	#
x_valid, ytime_valid, yevent_valid = load_BRCA(f"/Data/ljy/single_cox250/{name}_vaild_250.csv", dtype)
x_test, ytime_test, yevent_test = load_BRCA(f"/Data/ljy/single_cox250/{name}_test_250.csv", dtype)

opt_l2_loss = 0
opt_lr_loss = 0
opt_loss = torch.Tensor([float("Inf")])
###if gpu is being used
if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()
###
opt_c_index_va = 0
opt_c_index_tr = 0

###grid search the optimal hyperparameters using train and validation data
for l2 in L2_Lambda:
	for lr in Initial_Learning_Rate:
		loss_train, loss_valid, c_index_tr, c_index_va , _ = trainnet(x_train, ytime_train, yevent_train, \
																x_valid, ytime_valid, yevent_valid, \
																feature_size, class_size, latent_size, \
																lr, l2, num_epochs, name)
		if loss_valid < opt_loss:
			opt_l2_loss = l2
			opt_lr_loss = lr
			opt_loss = loss_valid
			opt_c_index_tr = c_index_tr
			opt_c_index_va = c_index_va
		print ("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_valid,"opt_c_index_tr:",opt_c_index_tr,"opt_c_index_va:",opt_c_index_va)



###train CVaDeS with optimal hyperparameters using train data, and then evaluate the trained model with test data
###Note that test data are only used to evaluate the trained CVaDeS
loss_train, loss_test, c_index_tr, c_index_te, evel_pred = trainnet(x_train, ytime_train, yevent_train, \
							x_test, ytime_test, yevent_test, \
							feature_size, class_size, latent_size, \
							opt_lr_loss, opt_l2_loss, Num_EPOCHS, name)

km_logrank_picture(evel_pred.cpu(),ytime_test.cpu(),yevent_test.cpu())
print ("Optimal L2: ", opt_l2_loss, "Optimal LR: ", opt_lr_loss)
print("C-index in Test: ", c_index_te)
