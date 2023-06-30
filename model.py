import torch
import torch.nn as nn
import torch.nn.functional as F
from Survival_CostFunc_CIndex import neg_par_log_likelihood , rank_loss
device_id = 0
torch.cuda.set_device(device_id)
class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(feature_size + class_size, 25)
        self.fc2 = nn.Linear(25, 5)

        self.fc2_mu = nn.Linear(5, latent_size)
        self.fc2_log_std = nn.Linear(5, latent_size)
        
        self.fc3 = nn.Linear(latent_size + feature_size, 25)
        self.fc4 = nn.Linear(25, 10) # feature_size
        self.fc5 = nn.Linear(10, 1)

    def encode(self, x, y):
        h1 = F.tanh(self.fc1(torch.cat([x, y], dim=1)))  # concat features and labels   
        h1 = F.tanh(self.fc2(h1))

        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std
    def decode(self, z, y):
        h3 = F.tanh(self.fc3(torch.cat([z, y], dim=1)))  # concat latents and labels    
        h3 = F.tanh(self.fc4(h3))
        recon = torch.sigmoid(self.fc5(h3))  
        return recon
    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z
    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        #z = torch.ones(x.shape[0],10).cuda()
        pred = self.decode(z, x)
        return pred, mu, log_std,z
    def loss_function(self, pred, ytime, yevent, mu, log_std) -> torch.Tensor:
        negloss=neg_par_log_likelihood(pred, ytime, yevent)
        rankloss=rank_loss(pred, ytime, yevent)
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        
        loss = negloss + 0.25*kl_loss + 0.5*rankloss
        return loss

