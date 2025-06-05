import random, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation import gen_data_sphere_cubic
import scipy
import scipy.io


N = 100
V = [64, 64, 8]
P = 3
Q = 2


def seed_everything(seed: int):

    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(1000)


def MSE(A, B):
    mse = ((A - B)**2).mean()
    return mse

response, A, beta, delta, Z, noise, eta = gen_data_sphere_cubic(N, V, P, Q, corr=1, scale=0.5, orthogonal=True)

X = np.hstack((np.ones((N,1)), A))
y_sm = np.zeros((N,np.prod(V)))
for n in range(N):
    y_sm[n] = scipy.ndimage.gaussian_filter(response[n].reshape(V), sigma=1, truncate=10).flatten()
    
#mask = np.argwhere(np.absolute(response[0])>0.001)
    
#%%
beta_voxels_3d_0 = beta[1].reshape(V)

i=4
beta_voxels_2d_i = beta_voxels_3d_0[:,:,i]
beta_imgplot_i = plt.imshow(beta_voxels_2d_i)
plt.colorbar()
plt.show()

beta_voxels_3d_0 = beta[2].reshape(V)
beta_voxels_2d_i = beta_voxels_3d_0[:,:,i]
beta_imgplot_i = plt.imshow(beta_voxels_2d_i)
plt.colorbar()
plt.show()


beta_voxels_3d_0 = beta[3].reshape(V)
beta_voxels_2d_i = beta_voxels_3d_0[:,:,i]
beta_imgplot_i = plt.imshow(beta_voxels_2d_i)
plt.colorbar()
plt.show()


delta_voxels_3d_0 = delta[0].reshape(V)
delta_voxels_2d_0 = delta_voxels_3d_0[:,:,i]
delta_imgplot_0 = plt.imshow(delta_voxels_2d_0)
plt.colorbar()
plt.show()

delta_voxels_3d_0 = delta[1].reshape(V)
delta_voxels_2d_0 = delta_voxels_3d_0[:,:,i]
delta_imgplot_0 = plt.imshow(delta_voxels_2d_0)
plt.colorbar()
plt.show()


eta_voxels_3d_0 = eta[0].reshape(V)
eta_voxels_2d_0 = eta_voxels_3d_0[:,:,i]
eta_imgplot_0 = plt.imshow(eta_voxels_2d_0)
plt.colorbar()
plt.title('eta_subject_1')
plt.show()

X = np.hstack((np.ones((N,1)), A))
truth =  X @ beta + Z @ delta
response_voxels_3d_0 = truth[0].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()
plt.title('y_without_eta_error_subject1')
plt.show()

response_voxels_3d_0 = truth[1].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()
plt.title('y_without_eta_error_subject2')
plt.show()

response_voxels_3d_0 = truth[2].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()
plt.title('y_without_eta_error_subject3')
plt.show()

response_voxels_3d_0 = response[0].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()

plt.show()

response_voxels_3d_0 = response[1].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()

plt.show()

response_voxels_3d_0 = response[2].reshape(V)
response_voxels_2d_0 = response_voxels_3d_0[:,:,i]
response_imgplot_0 = plt.imshow(response_voxels_2d_0)
plt.colorbar()

plt.show()


#%%
print('Running DISRM without threshold')
from DISRM_HC import DISRM
beta_est, sv_est, he_est = DISRM(covariates=A, images=response, images_sm=y_sm, img_shape=V, q=Q,
        learning_rate1 = 0.002, batch_size1=64, epochs1 = 250, cuda=False, mask=None, threshold = None)


response_est = X @ beta_est+ sv_est @ he_est

print('mse_hidden', MSE(sv_est@he_est, Z@delta))
print('mse_beta', MSE(beta_est, beta))
print('mse_response', MSE(response_est, truth))

#%%
print('Running DISRM with threshold')
def compute_threshold(covariates, images, n_permute, V, Q):
    n_observations = covariates.shape[0]
    beta_null_list = []
    for i in range(n_permute):
        order = np.random.choice(
                np.arange(n_observations), n_observations,
                replace=False)
        shuffled = covariates[order]
        beta_null, _, _ = DISRM(covariates=shuffled, images=response, images_sm=y_sm, img_shape=V, q=Q,
                learning_rate1 = 0.002, batch_size1=64, epochs1 = 250, cuda=False, mask=None, threshold = None)
        beta_null_list.append(beta_null)
        
    beta_null = np.stack(beta_null_list, 1)
    threshold = np.quantile(np.abs(beta_null), 0.05, (1))
    return threshold

threshold = compute_threshold(A, response, 20, V, Q)


beta_est_th, sv_est_th, he_est_th = DISRM(covariates=A, images=response, images_sm=y_sm, img_shape=V, q=Q,
        learning_rate1 = 0.002, batch_size1=64, epochs1 = 250, threshold = threshold, cuda=False, mask=None)

response_est_th = X @ beta_est_th+ sv_est_th @ he_est_th

print('mse_hidden', MSE(sv_est_th@he_est_th, Z@delta))
print('mse_beta', MSE(beta_est_th, beta))
print('mse_response', MSE(response_est_th, truth))
 
#%%
i=4
fig, axes = plt.subplots(nrows=2, ncols=3)
fig.tight_layout(h_pad=0.1, w_pad=0.1)
ax1, ax3, ax5, ax2, ax4, ax6 = axes.flatten()
im1=ax1.imshow(beta[1].reshape(V)[:,:,i])
ax1.set_title('beta1')
im2=ax2.imshow(beta_est[1].reshape(V)[:,:,i])
ax2.set_title('beta1_DISRM')
fig.colorbar(im1, ax=[ax1, ax2], shrink=0.4)

im3=ax3.imshow(beta[2].reshape(V)[:,:,i])
ax3.set_title('beta2')
im4=ax4.imshow(beta_est[2].reshape(V)[:,:,i])
ax4.set_title('beta2_DISRM')
fig.colorbar(im3, ax=[ax3,ax4], shrink=0.4)

im5=ax5.imshow(beta[3].reshape(V)[:,:,i])
ax5.set_title('beta3')
im6=ax6.imshow(beta_est[3].reshape(V)[:,:,i])
ax6.set_title('beta3_DISRM')
fig.colorbar(im5, ax=[ax5,ax6], shrink=0.4)

plt.show()
plt.close()


fig, axes = plt.subplots(nrows=2, ncols=3)
fig.tight_layout(h_pad=0.1, w_pad=0.1)
ax1, ax3, ax5, ax2, ax4, ax6 = axes.flatten()
im1=ax1.imshow(truth[0].reshape(V)[:,:,i])
ax1.set_title('truth')
im2=ax2.imshow(response_est[0].reshape(V)[:,:,i])
ax2.set_title('reconstruct')
fig.colorbar(im1, ax=[ax1, ax2], shrink=0.4)

im3=ax3.imshow(truth[1].reshape(V)[:,:,i])
ax3.set_title('truth')
im4=ax4.imshow(response_est[1].reshape(V)[:,:,i])
ax4.set_title('reconstruct')
fig.colorbar(im3, ax=[ax3,ax4], shrink=0.4)

im5=ax5.imshow(truth[2].reshape(V)[:,:,i])
ax5.set_title('truth')
im6=ax6.imshow(response_est[2].reshape(V)[:,:,i])
ax6.set_title('reconstruct')
fig.colorbar(im5, ax=[ax5,ax6], shrink=0.4)

plt.show()
plt.close()


delta_voxels_3d_0 = he_est[0].reshape(V)
delta_voxels_2d_0 = delta_voxels_3d_0[:,:,i]
delta_imgplot_0 = plt.imshow(delta_voxels_2d_0)
plt.colorbar()
plt.title('hidden_effect_1_est')
plt.show()

delta_voxels_3d_0 = he_est[1].reshape(V)
delta_voxels_2d_0 = delta_voxels_3d_0[:,:,i]
delta_imgplot_0 = plt.imshow(delta_voxels_2d_0)
plt.colorbar()
plt.title('hidden_effect_2_est')
plt.show()
plt.close()