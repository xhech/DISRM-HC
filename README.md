# **Deep Image-on-scalar Regression with Hidden Confounders**

DISRM-HC simultaneously detects hidden confounders through functional surrogate variable analysis and learns primary effects from variables of interest by neural networks.

## Code Description
### DISRM_HC.py
> DSIRM_HC model architecture
### simulation.py
> simulation of covariates, primary effects, hidden confounders, hidden effects, subject-specific variability, random noise and image responses.

### sim_DISRM.py
> simulation and visualization; run DISRM-HC for estimation
>
examples:
```python
from simulation import gen_data_sphere_cubic
response, A, beta, delta, Z, noise, eta = gen_data_sphere_cubic(N, V, P, Q, corr = 1, scale = 0.5, orthogonal=True)
```
```python
from DISRM_HC import DISRM
beta_est, surrogate_est, hidden_effect_est = DISRM(covariates, images, images_sm, img_shape, q,
                                                   learning_rate1, batch_size1, epochs1,
                                                   cuda = False, mask=None, threshold = None)
```
