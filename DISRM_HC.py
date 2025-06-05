#%%
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Network(nn.Module):
    def __init__(self,in_dim,out,out_1,out_2):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out)
        )
        self.a = nn.Linear(out,out_1)
        self.b = nn.Linear(out,out_2)

        
    def forward(self, x):
        
        common = self.shared_layer(x) #x:(B,C,D)
        beta_out = self.a(common) #(B,C,P)
        delta_out = self.b(common) #(B,C,Q)
    
        return beta_out, delta_out
    

def loss_function(coefficient_beta, covariate,coefficient_gamma,confounders, response, device):
    he = torch.Tensor(coefficient_gamma).to(device)
    X = torch.hstack((torch.ones((covariate.shape[0],1)).to(device), covariate.to(device)))
    loss = torch.mean(torch.square(response.to(device) - X @ coefficient_beta.T
        -confounders.to(device) @ he.T))
    return loss


def train(coordinates, target, covariate, sv, in_dims,out_dims,
 out_1, out2, learning_rate, batch_size, epochs, threshold, cuda):
    
    if cuda==True:
        device = 'cuda'
    else:
        device ='cpu'
        
    model = Network(in_dims, out_dims,out_1, out2)
    model.to(device)
    
    if threshold is not None:
        training_data = TensorDataset(coordinates, target, threshold)
    else:
        training_data = TensorDataset(coordinates, target)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    #num_train_steps = len(training_data)/batch_size
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps*epochs)

    model.train()
    total_loss_train_history = []
    for epoch in range(epochs):
        total_loss_train = []
        if threshold is not None:
            for batch_idx, (train_feature, train_target, threshold_batch) in enumerate(train_dataloader):
                train_feature = train_feature.to(device)
                train_target = train_target.to(device)
                threshold_batch = threshold_batch.to(device)
                
                optimizer.zero_grad()
                
                coefficient_beta, coefficient_delta = model(train_feature, cuda)
                
                #apply threshold
                coefficient_beta = coefficient_beta.abs()*(coefficient_beta.abs()>threshold_batch)*torch.sign(coefficient_beta)
                
                loss = loss_function(coefficient_beta,covariate, coefficient_delta, sv, train_target.T, device)
                
                total_loss_train.append(loss.item())
    
                loss.backward()
                optimizer.step()
        else:
            for batch_idx, (train_feature, train_target) in enumerate(train_dataloader):
                train_feature = train_feature.to(device)
                train_target = train_target.to(device)
                
                optimizer.zero_grad()
                
                coefficient_beta, coefficient_delta = model(train_feature, cuda)
                
                loss = loss_function(coefficient_beta,covariate, coefficient_delta, sv, train_target.T, device)
                
                total_loss_train.append(loss.item())
    
                loss.backward()
                optimizer.step()
                #scheduler.step()
            
        loss_epoch = np.mean(total_loss_train)
        total_loss_train_history.append(loss_epoch)
            
        if (epoch+1)%50 == 0:
            print(f"loss: {loss_epoch:>7f}  [{epoch+1:>5d}/{epochs:>5d}]")
        
    return model, total_loss_train_history



def predict(feature_tensor, model, cuda):
    if cuda==True:
        device = 'cuda'
    else:
        device = 'cpu'
    with torch.no_grad():
        model = model.to(device) 
        model.eval()
        beta_pred, delta_pred = model(feature_tensor.to(device), cuda)
    
    return beta_pred, delta_pred

def get_coordinates(dim):
    d = np.prod(dim)
    coords = np.c_[np.unravel_index(np.arange(d).reshape(-1, 1), dim)]
    coords = coords / coords.max(0, keepdims=True)
    return coords
    
def get_masked_coords(mask, img_shape):
    coords_raw = np.c_[np.unravel_index(mask, img_shape)]
    coords = coords_raw / coords_raw.max(0, keepdims=True)
    return coords

   
def DISRM_uni(covariates, images, images_sm, img_shape, q,
        learning_rate1, batch_size1, epochs1, mask, threshold, cuda):
    
        n_images, n_covariates = covariates.shape
            
        if mask is None:
            coordinates = get_coordinates(img_shape)
        else:
            coordinates = get_masked_coords(mask, img_shape)
        
        img_size = coordinates.shape[0]

        print('Learning beta_star...')
        
        coordinates = torch.Tensor(coordinates)
        covariates = torch.Tensor(covariates)     
        images = torch.Tensor(images)    
        images_sm = torch.Tensor(images_sm)
        
        if threshold is not None:
            threshold = torch.Tensor(threshold).T
        
        X = torch.hstack((torch.ones((covariates.shape[0],1)), covariates))
        beta_star = torch.linalg.lstsq(X, images_sm).solution
        
        print('Computing SVD...')
        
        resid = images- X @ beta_star
        U,D,V = torch.svd(resid)
        Uq = U[:,:q]
  
        print('learning gamma')
        t0 = time()
        gamma = torch.linalg.lstsq(Uq, resid).solution
        Proj = torch.eye(img_size) - torch.ones([1,img_size]).T @ torch.ones([1,img_size]) /img_size
        bpp = beta_star @ Proj @gamma.T @ torch.linalg.pinv(gamma @ Proj @ gamma.T)
        g_mat = Uq+ X @ bpp
        # gamma = torch.linalg.inv(Uq.T@Uq)@Uq.T@resid
        # b_gamma = beta_star@gamma.T
        # g_mat = Uq+ X @ b_gamma @ torch.linalg.inv(gamma@gamma.T)
        
        
        print('Finished in {} sec'.format(int(time() - t0)))
        
    
        print('Learning beta...')
        t0 = time()
            
        model_coef, total_loss_train_history = train(coordinates = coordinates, target = images.T, covariate = covariates, sv = g_mat,
                                 in_dims = coordinates.shape[1], out_dims=8, out_1= covariates.shape[1]+1, out2 = q,
                                 learning_rate = learning_rate1, batch_size = batch_size1, epochs = epochs1, threshold = threshold, cuda=cuda)
        
        beta, psi = predict(coordinates, model_coef, cuda)
        
        if threshold is not None:
            beta = beta.abs()*(beta.abs()>threshold)*torch.sign(beta)
            
        print('Plotting Loss for model1...')
        plt.plot(list(range(1, epochs1+1)), total_loss_train_history, label='train_total')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        plt.savefig('loss_history.png')
        plt.close()
            
        print('Finished in {} sec'.format(int(time() - t0)))

        beta_np = beta.detach().cpu().numpy().T
        g_mat_np = g_mat.detach().cpu().numpy()
        #gamma_np = gamma.detach().cpu().numpy()
        psi_np = psi.detach().cpu().numpy().T
        
        if mask is None:
            beta_full = beta_np
            hid_full = psi_np
        
        else:
            beta_full = np.zeros((covariates.shape[1]+1, np.prod(img_shape)))
            hid_full = np.zeros((q, np.prod(img_shape)))     
            beta_full[:, mask.squeeze()] = beta_np
            hid_full[:, mask.squeeze()] = psi_np
        
        return beta_full, g_mat_np, hid_full
    
    
def DISRM_multi(covariates, images, images_sm, img_shape, q, learning_rate1, batch_size1, epochs1, mask, threshold, cuda):
    beta_all = []
    sv_all = []
    he_all = []
    for j in range(images.shape[0]):
        beta, sv, he = DISRM_uni(covariates, images[j,:,:], images_sm[j,:,:], img_shape, q,
                learning_rate1, batch_size1, epochs1, mask, threshold, cuda)
        beta_all.append(beta)
        sv_all.append(sv)
        he_all.append(he)
        
    beta_all = np.stack(beta_all)
    sv_all = np.stack(sv_all)
    he_all = np.stack(he_all)
    
    return beta_all, sv_all, he_all


def DISRM(covariates, images, images_sm, img_shape, q, learning_rate1, batch_size1, epochs1, mask = None, threshold = None, cuda=True):
    """

    Parameters
    ----------
    covariates : numpy array
        shape (N,P).
    images : flattened numpy array of imaging responses
        shape (N, M) or (C, N, M).
    images_sm : flattened numpy array of smoothed imaging responses
        shape (N, M) or (C, N, M).
    img_shape : list
        list of integers such as [32,32,32].
    q : integer
        number of hidden factors.
    learning_rate1 : float
    batch_size1 : integer
    epochs1 : integer
    mask : numoy array, optional
        image mask. The default is None.
    cuda : bool, optional
        use gpu. The default is True.

    Returns
    -------
    beta_all : primary effect
        shape (P+1, M) or (C, P+1, M). With intercept at index 0.
    sv_all : surrogate variable
        shape (N, Q).
    he_all : hidden effect
        shape (Q, M) or (C, Q, M).

    """
    if len(images.shape)==2:
        beta_all, sv_all, he_all = DISRM_uni(covariates, images, images_sm, img_shape, q,
                learning_rate1, batch_size1, epochs1, mask, threshold, cuda)
    if len(images.shape)==3:
        beta_all, sv_all, he_all = DISRM_multi(covariates, images, images_sm, img_shape, q,
                learning_rate1, batch_size1, epochs1, mask, threshold, cuda)
    return beta_all, sv_all, he_all
    
    
