import torch
import torch.nn as nn
import math
import numpy as np
class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end, device=torch.device('cpu')):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.betas=torch.linspace(beta_start, beta_end, num_timesteps) #linear schedule going from beta_start to beta_end
        self.alphas=1-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod=torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alpha_cumprod=torch.sqrt(1-self.alphas_cumprod)
        
        # Move tensors to the specified device
        self.to(device)
    
    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self

    #forward process
    def add_noise(self, originial, noise, t):
        # Ensure t is on the correct device
        t = t.to(self.device)
        
        original_shape=originial.shape
        batch_size=original_shape[0]
        sqrt_alpha_cumprod_t=self.sqrt_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        noisy_image=sqrt_alpha_cumprod_t*originial+sqrt_one_minus_alpha_cumprod_t*noise
        return noisy_image
    
    #reverse process
    def sample_prev_timestep(self, xt, t, noise_pred):
        # Ensure t is on the correct device
        t = t.to(self.device)
        
        # Get batch size for reshaping
        batch_size = xt.shape[0]
        
        # Reshape tensor coefficients for proper broadcasting
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(batch_size, 1, 1, 1)
        betas_t = self.betas[t].reshape(batch_size, 1, 1, 1)
        alphas_t = self.alphas[t].reshape(batch_size, 1, 1, 1)
        
        # Calculate x0 using reshaped tensors for proper broadcasting
        x0 = (xt - (sqrt_one_minus_alpha_cumprod_t * noise_pred)) / sqrt_alpha_cumprod_t
        x0 = torch.clamp(x0, -1., 1.)

        # Calculate mean using reshaped tensors
        mean = xt - ((betas_t * noise_pred) / sqrt_one_minus_alpha_cumprod_t)
        mean = mean / torch.sqrt(alphas_t)

        if t[0] == 0:  # Assuming all elements in t are the same
            return mean, x0
        else:
            # Handle t-1 case with proper reshaping
            t_prev = t - 1
            alphas_cumprod_t = self.alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
            alphas_cumprod_t_prev = self.alphas_cumprod[t_prev].reshape(batch_size, 1, 1, 1)
            
            variance = (betas_t * (1 - alphas_cumprod_t_prev)) / (1 - alphas_cumprod_t)
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, x0

    
class CosineNoiseScheduler:
    def __init__(self, num_timesteps,  max_beta=0.999):
        self.betas = []
        self.num_timesteps = num_timesteps
        self.max_beta = max_beta
        
    
    def get_betas(self):  
        for i in range(self.num_timesteps):
            t1 = i/self.num_timesteps
            t2 = (i+1)/self.num_timesteps
            alpha_bar_t1= math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar_t2 = math.cos((t2+0.008)/1.008 * math.pi/2)**2
            self.betas.append(min(1-alpha_bar_t1/alpha_bar_t2, self.max_beta))
            
        self.betas = np.array(self.betas)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self
    
    
    
      
    
    
    
        
        
            