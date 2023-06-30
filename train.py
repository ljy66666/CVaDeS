
from model import CVAE
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index,rank_loss
from km_logrank import km_logrank_picture


device_id = 0
torch.cuda.set_device(device_id)
dtype = torch.FloatTensor

def trainnet(train_x, train_ytime, train_yevent, \
			eval_x, eval_ytime, eval_yevent, \
			feature_size, class_size, latent_size, \
			Learning_Rate, L2, Num_Epochs, name):            
	
    net = CVAE(feature_size, class_size, latent_size)
    ###if gpu is being used
    if torch.cuda.is_available():
        net.cuda()
    ###
    ###optimizer
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)

    for epoch in range(Num_Epochs+1):
        net.train()
        opt.zero_grad() ###reset gradients to zeros
       
        pred, mu, log_std,train_z = net(train_x, train_ytime) ###Forward
        loss=net.loss_function(pred,train_ytime, train_yevent, mu, log_std)
        loss.backward() ###calculate gradients
        opt.step() ###update weights and biases       

        if epoch % 10 == 0: 
                   
            net.eval()
            
            eval_pred=net.decode(train_z[:eval_x.shape[0],:],eval_x)
            eval_loss=net.loss_function(eval_pred,eval_ytime, eval_yevent,mu[:eval_x.shape[0],:],log_std[:eval_x.shape[0],:])

            train_cindex = c_index(pred, train_ytime, train_yevent)
            eval_cindex = c_index(eval_pred, eval_ytime, eval_yevent)
            print("Loss in Train: ", loss)

    return (loss, eval_loss, train_cindex, eval_cindex,eval_pred)