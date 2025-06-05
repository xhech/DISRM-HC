import numpy as np


def gen_noise_var(N, V):
    var = 1 / np.random.gamma(shape=10, scale=1/9, size=(N,V))
    return var

def gen_noise(N, V, scale):
    var = gen_noise_var(N, V)
    noise = np.random.normal(0, var)/scale
    return noise

def gen_response(A, beta, Z, delta, eta, noise):
    Y = A@beta + Z@delta + eta + noise
    return Y
 
    
def choose_cubic(V):
    
    size1=[int(V[0]/3.2),int(V[1]/3.2),V[2]-1]
    size2=[int(V[0]/3.2)-1,int(V[1]/3.2),V[2]-1]
    
    if V==[64,64,8]:
        start1 = [9, 15, 0]
        start2 = [40, 35, 0]
        coords1 = []
        coords2 = []
        for i in range(len(V)):
            coords1.append(np.arange(start1[i], start1[i]+size1[i]))
            coords2.append(np.arange(start2[i], start2[i]+size2[i]))
        
        
    else:
        start1=[]
        start2=[]
        coords1 = []
        coords2=[]
        for i in range(len(V)):
            start1.append(np.random.randint(low=0, high=V[i]-size1[i]))
            coords1.append(np.arange(start1[i], start1[i]+size1[i]))
            
            start2.append(np.random.randint(low=0, high=V[i]-size2[i]))
            coords2.append(np.arange(start2[i], start2[i]+size2[i]))
        
    return coords1, coords2

def get_coords_flat(V):
    coords_flat1 = np.ravel_multi_index(np.meshgrid(choose_cubic(V)[0][0], choose_cubic(V)[0][1], choose_cubic(V)[0][2]), (V[0],V[1],V[2])).flatten()
    coords_flat2 = np.ravel_multi_index(np.meshgrid(choose_cubic(V)[1][0], choose_cubic(V)[1][1], choose_cubic(V)[1][2]), (V[0],V[1],V[2])).flatten()
    
    return coords_flat1, coords_flat2


def sphere_centers(V):
    
    center1 = [int(V[0]/4), int(V[1]/4), int(V[2]/2)]
    center2 = [int(V[0]*3/4), int(V[1]*3/4), int(V[2]/2)]
    center3 = [int(V[0]/4), int(V[1]*3/4), int(V[2]/2)]
    center4 = [int(V[0]*3/4), int(V[1]/4), int(V[2]/2)]
    
    return center1, center2, center3, center4
    
               

def choose_sphere_cubic(V, P=3, Q=2, R=12):
    
    center1, center2, center3, center4 = sphere_centers(V)
    
    
    beta = np.zeros([V[0],V[1],V[2],P])
    delta = np.zeros([V[0],V[1],V[2],Q])
    
    div = R/6
    
    for p in range(P):
        for q in range(Q):
            for i in range(V[0]):
                for j in range(V[1]):
                    for k in range(V[2]):
                        
                        dist1 = np.linalg.norm(np.array([i,j,k]) - center1)
                        dist2 = np.linalg.norm(np.array([i,j,k]) - center2)
                        dist3 = np.linalg.norm(np.array([i,j,k]) - center3)
                        dist4 = np.linalg.norm(np.array([i,j,k]) - center4)
                        
                        if p == 0:
                            if dist1<=R:
                                beta[i,j,k,p] = (R - dist1)/(2*div)
                        if p == 1:
                            if dist2<=R:
                                beta[i,j,k,p] = 1 - np.exp((R - dist2)/(4.5*div))
                                       
                        
                        if q==0:                     
                            if dist3<=R/2:
                                delta[i,j,k,q] = (R/2 - dist3)/3
                        if q>0: 
                            if dist4<=R:
                                delta[i,j,k,q] = dist4/6
                                
    for i in range(int(V[0]/2), int(V[0]/2)+2):
        for j in range(V[1]):
            for k in range(V[2]):
                beta[i,j,k,0] = 1.5
                
    for i in range(int(V[0]/2), V[0]):
        for j in range(int(V[1]/2), int(V[1]/2)+2):
            for k in range(V[2]):
                beta[i,j,k,1] = 1 + (i+j+k)/(V[0]+V[1]+V[2])
                                
    beta_flat = []
    for i in range(P):
        beta_flat.append(beta[:,:,:,i].flatten())
    beta = np.array(beta_flat)       
    
    for p in range(P):
        if p>=2:
            coords_flat1, _ = get_coords_flat(V)
            for j in range(len(coords_flat1)):
                beta[p, coords_flat1[j]] =  coords_flat1[j]/coords_flat1.max() 
                     
    beta0 = np.zeros((1, np.prod(V)))
    beta = np.vstack((beta0, beta))
    
    delta_flat = []
    for i in range(Q):
        delta_flat.append(delta[:,:,:,i].flatten())
    delta = np.array(delta_flat)     

    return  beta, delta

def choose_sphere_cubic2(V, P=3, Q=2, R=12):
    
    center1, center2, center3, center4 = sphere_centers(V)
    
    
    beta = np.zeros([V[0],V[1],V[2],P])
    delta = np.zeros([V[0],V[1],V[2],Q])
    
    div = R/6
    
    for p in range(P):
        for q in range(Q):
            for i in range(V[0]):
                for j in range(V[1]):
                    for k in range(V[2]):
                        
                        dist1 = np.linalg.norm(np.array([i,j,k]) - center1)
                        dist2 = np.linalg.norm(np.array([i,j,k]) - center2)
                        dist3 = np.linalg.norm(np.array([i,j,k]) - center3)
                        dist4 = np.linalg.norm(np.array([i,j,k]) - center4)
                        
                        if p == 0:
                            if dist1<=R:
                                beta[i,j,k,p] = dist1/(3*div)
                        if p == 1:
                            if dist2<=R:
                                beta[i,j,k,p] = np.exp((R - 1.5*dist2)/(4*div))
                                       
                        
                        if q==0:                     
                            if dist3<=R/2:
                                delta[i,j,k,q] = dist3**2 / 12
                        if q>0: 
                            if dist4<=R:
                                delta[i,j,k,q] = 2 -np.exp(dist4/6)
                                
    for i in range(int(V[0]/2), int(V[0]/2)+2):
        for j in range(V[1]):
            for k in range(V[2]):
                beta[i,j,k,0] = 1
                
    for i in range(int(V[0]/2), V[0]):
        for j in range(int(V[1]/2), int(V[1]/2)+2):
            for k in range(V[2]):
                beta[i,j,k,1] = 0.5 + np.exp((i+j+k)/(V[0]+V[1]+V[2]))
                                
    beta_flat = []
    for i in range(P):
        beta_flat.append(beta[:,:,:,i].flatten())
    beta = np.array(beta_flat)       
    
    for p in range(P):
        if p>=2:
            coords_flat1, _ = get_coords_flat(V)
            for j in range(len(coords_flat1)):
                beta[p, coords_flat1[j]] =  0.8 - coords_flat1[j]/coords_flat1.max()
                     
    beta0 = np.zeros((1, np.prod(V)))
    beta = np.vstack((beta0, beta))
    
    delta_flat = []
    for i in range(Q):
        delta_flat.append(delta[:,:,:,i].flatten())
    delta = np.array(delta_flat)     

    return  beta, delta

    
def gen_data_sphere_cubic(N, V, P, Q, corr, scale, orthogonal=True):
    
    A2 = np.zeros((N,1))
    index = np.random.choice(N, int(N/2), replace=False)  
    A2[index] = 1
    if P>1:
        A1 = np.random.randn(N, P-1) + 1
        A = np.hstack((A1, A2))
    if P==1:
        A = np.random.randn(N, 1) + 1
    
    Z1_1 = np.random.normal(loc=0.5, scale=0.5, size=(int(N/2), 1))
    Z1_2 = np.random.normal(loc=-0.5, scale=0.5, size=(int(N/2), 1))
    Z1 = np.vstack((Z1_1, Z1_2))
    Z2 = np.random.binomial(1, 0.5, size=(N,1))
 
    if corr==0:
        G = np.zeros((P,1))
    if corr==1:
        G = np.random.uniform(0, 0.2, size=(P,1))
    if corr==2:
        G = np.random.uniform(0.2, 0.5, size=(P,1))
    if corr==3:
        G = np.random.uniform(0.5, 1, size=(P,1))
    B = np.random.binomial(1,0.5,size = (P,1))
    for i in range(P):
        G[i]=G[i]*(2*B[i]-1)
    Z4 = A@G  + np.random.normal(loc=0, scale=0.5, size=(N,1))

    if Q==1:
        Z = Z4
    if Q==2:
        Z = np.hstack((Z4,Z1))
    if Q==3: 
        Z = np.hstack((Z4,Z1,Z2))
    if Q>3:
        Z3 = np.random.normal(loc=0.5, scale=0.5, size=(N,Q-3))
        Z = np.hstack((Z4,Z1,Z2,Z3))


    VV = np.prod(V)
    
    beta, delta = choose_sphere_cubic(V, P, Q, R=12)
    #beta2, delta2 = choose_sphere_cubic2(V, P, Q, R=12)
   
    if orthogonal is not True:
        delta = np.random.normal(size = (Q, VV), loc=1, scale=0.25)
        #delta2 = np.random.normal(size = (Q, VV), loc=1, scale=0.25)

    
    noise = gen_noise(N, VV, scale)
    #noise2 = gen_noise(N, VV, scale)

    all_coords = np.arange(0, VV)
    s = all_coords/len(all_coords)
    eigenfunc1 = 2*s-1
    eigenfunc2 = s-0.5
    
    eigenfunc1 = np.reshape(eigenfunc1, (1,-1))
    eigenfunc2 = np.reshape(eigenfunc2, (1,-1))
    
    
    xi1 = np.random.normal(loc=0, scale=0.1, size=(N,1))
    xi2 = np.random.normal(loc=0, scale=0.1, size=(N,1))
    
    #xi3 = np.random.normal(loc=0, scale=0.15, size=(N,1))
    #xi4 = np.random.normal(loc=0, scale=0.15, size=(N,1))
    
    eta = xi1@eigenfunc1 + xi2@eigenfunc2
    #eta2 = xi3@eigenfunc1 + xi4@eigenfunc2
    
    #eta = np.zeros((N,VV))
    #noise = np.zeros((N,VV))
    
    #eta2 = np.zeros((N,VV))
    #noise2 = np.zeros((N,VV))
     
    X = np.hstack((np.ones((N,1)), A))
    response = gen_response(X, beta, Z, delta, eta, noise)
    #response2 = gen_response(X, beta2, Z, delta2, eta2, noise2)
    
    #response = np.stack((response1, response2), axis=2)
    #beta = np.stack((beta, beta2), axis=2)
    #delta = np.stack((delta, delta2), axis=2)
    #noise = np.stack((noise, noise2), axis=2)
    #eta = np.stack((eta, eta2), axis=2)
    
    # response = np.stack((response, response2))
    # beta = np.stack((beta1, beta2))
    # delta = np.stack((delta1, delta2))
    # noise = np.stack((noise1, noise2))
    # eta = np.stack((eta1, eta2))

    return response, A, beta, delta, Z, noise, eta

