from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0 #TODO : aca hacen algo diferente con non zero mask... 
    # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    # return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=True,
        action_weight=1.0, loss_discount=1.0,p_mask=0.5):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim+1 # (S+A+R)
        self.model = model
        self.action_weight=action_weight
        self.loss_discount=loss_discount

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type](self.action_dim)

        self.bernoulli_dist = torch.distributions.Bernoulli(p_mask)


    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            
        '''
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    


    def p_mean_variance(self,x, past_K_history, K_step, t, returns=None):
        # TODO falta agregar que el modelo puede usar otras condiciones...  
        if self.returns_condition:  # de donde sale eso... TODO 
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, past_K_history=past_K_history, K_step=K_step, t=t, returns=returns, use_dropout=False)
            epsilon_uncond = self.model(x, past_K_history=past_K_history, K_step=K_step, t=t, returns=returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, past_K_history=past_K_history, K_step=K_step, t=t, returns=returns, force_dropout=True)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    
    @torch.no_grad()
    def p_sample(self, x, past_K_history,K_step, t, returns=None): # Falta REpaint sampling TODO
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, past_K_history=past_K_history, K_step=K_step, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, past_K_history, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        """ Classical DDPM (check this) sampling algorithm with sample_fn incorporated and conditioning
        
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) # TODO:  en dd usan 0.5*torch.randn(shape, device=device) y no en mtdiff
        #x = apply_conditioning(x, cond, self.action_dim) # apply conditioning to first sample 

        chain = [x] if return_chain else None
        K_step=past_K_history.shape[1] # ver esto... TODO
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x = self.p_sample(self, x, past_K_history,K_step, t, **sample_kwargs) # sample with conditions
        #    x = apply_conditioning(x, cond, self.action_dim) # apply conditioning again # TODO: cambiar aca por el mask y todo y usar repaint sampling... 

            progress.update({'t': i})
            if return_chain: chain.append(x)

        progress.stamp() # en otros usar .close()
        # cambio aqui el sort b yvalues. es lo mismo solo q lo uso en la policy para tener mas control de lo que pasa y hacer um algoritmo mas general... 
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x,  chain)

    @torch.no_grad()
    def conditional_sample(self, past_K_history, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(past_K_history[0]) # TODO ver esto...  asume que el past k history esta batchified
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, past_K_history, **sample_kwargs) #   TODO add returns conditioning... 

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start,returns,K_step_mask, t):
        #TODO: OJO QUE ACA PARA MTDIFF usan unos einops, ver esto... 
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred_epsilon = self.model(x_noisy, returns, t, K=K_step_mask,use_dropout=True) # TODO: maybe add task and actual K step. ver modelo tambien

        assert noise.shape == pred_epsilon.shape
        loss_weights=self.get_mask_loss_weights(K_step_mask) # (B,)-> (B,H,T)
        loss, info = self.loss_fn(pred_epsilon, noise,loss_weights) # TODO: falta ver la funcion de loss... 

        return loss, info


    def loss(self, x, returns, *args): # maybe add task and actual K step.

        batch_size = len(x) # TODO aca mtdiff hace algo con einops...
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        K_step_mask = torch.randint(0, self.horizon, (batch_size,), device=x.device).long() # [0,H-1] #TODO test horizon... 
        samples = self.bernoulli_dist.sample((batch_size,))
        K_step_mask=samples*K_step_mask  # mask with certain probability
        # loss weights = torch.ones[:K_step_mask]=0

        return self.p_losses(x, returns, K_step_mask.long(), t) # TODO: maybe add task and actual K step.



    def forward(self, past_K_history, *args, **kwargs):
        return self.conditional_sample(past_K_history, *args, **kwargs) # TODO: repaint sampling... 





"""
 def get_mask_loss_weights(self,K_step_mask):
        '''
            sets loss coefficients for masked trajectory
            
            K_step_mask: (B,)
        returns: loss_weights: (B,H,T)
        '''
        batch_size=len(K_step_mask)
        dim_weights=torch.ones(batch_size, self.horizon, self.transition_dim,dtype=torch.float32)
        #dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        #discounts = self.loss_discount ** torch.arange(self.horizon-K, dtype=torch.float) #TODO: horizon es todo el horizonte y h-K es lo que hay q predecir... 
        #discounts = discounts / discounts.mean() # TODO: para usar los discount podemos calcularlos antes... y sumarlos al mask 

        #loss_weights = torch.einsum('h-k,t->(h-k)t', discounts, dim_weights)

        loss_weights[:,K_step_mask, :self.action_dim] = self.action_weight
        loss_weights[:,K_step_mask, self.action_dim:self.action_dim+self.observation_dim] = 0 # because the initial state is given 
        loss_weights[:,:K_step_mask, :] = 0 # mask behind

        return loss_weights

"""




"""
    def get_random_mask(self):
        #TODO samplear de una uniforme y mitad mitad... o quizas menos...
        raise NotImplementedError

"""


class GaussianDiffusion_task_rtg(nn.Module):
    """
    DDPM algorithm with 2 modes, task inference (unmasked plan) or policy sampling (masked plan and unmasked task) 
    """
    def __init__(self, model, horizon, observation_dim, action_dim,task_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=True,
        action_weight=1.0,rtg_weight=1.0, loss_discount=1.0,p_mode=0.5):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_dim=task_dim
        self.transition_dim = observation_dim + action_dim+1+task_dim+1 # (S+A+R+G+rtg)
        self.model = model
        self.action_weight=action_weight
        self.rtg_weight=rtg_weight
        self.loss_discount=loss_discount

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        self.bernoulli_dist = torch.distributions.Bernoulli(p_mode)

        self.loss_weights = self.get_loss_weights(action_weight,rtg_weight, loss_discount) # TODO 
        self.loss_fn = Losses[loss_type]()
        self.mask_dict=self.get_mask_mode()

    def get_loss_weights(self, action_weight, rtg_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t

        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)


        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight

        # always conditioning on s0
        loss_weights[0, self.action_dim:self.action_dim+self.observation_dim] = 0

        # manually set rtg weight
        loss_weights[0, -(self.task_dim+1)] = rtg_weight  # assumes A S R RTG TASK  
        return loss_weights.to(device="cuda").unsqueeze(0) # (1,H,T) TODO fix

    def get_mask_mode(self,device="cuda"): # assumes the following order: A S R RTG T 
        #TODO fix device 
        mask_dict={}

        # Crear el tensor de índices para la dimensión T
        indices = torch.arange(self.transition_dim).unsqueeze(0)  # Dimensiones (1, T)

        ### 0 mask ###

        mask_0 = (indices >= self.transition_dim-self.task_dim).float()  # (1, T)

        mask_0 = mask_0.expand(self.horizon, self.transition_dim).clone()  # (H, T)

        mask_0[0,self.action_dim:self.action_dim+self.observation_dim]= 1 # condition on s0
        mask_0=mask_0.to(device)
        mask_dict[0]=mask_0 # action inference 

        ### 1 mask ###

        # Comparar los índices con el umbral y crear la máscara
        mask_1 = (indices < self.transition_dim-self.task_dim).float()  # (1, T) -1 because the rtg is not a condition... TODO ver aca despues para el sampling si condicionar o no en el rtg..., no siempre esta disponible..  

        # Expandir la máscara a las dimensiones deseadas (H, T)
        mask_1 = mask_1.expand(self.horizon, self.transition_dim)  # (H, T)
        mask_1= mask_1.to(device)

        mask_dict[1]=mask_1 # task inference

        return(mask_dict)
    
    def get_mask_from_batch(self,mode_batch):
        stacked_values = torch.stack([self.mask_dict[0], self.mask_dict[1]])
        result_tensor = stacked_values[mode_batch]
        return(result_tensor)

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            
        '''
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    


    def p_mean_variance(self,x,t):

        epsilon = self.model(x=x, t=t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    
    @torch.no_grad()
    def p_sample(self, x, t): # Falta REpaint sampling TODO
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, verbose=True, return_chain=False):
        """ Classical DDPM (check this) sampling algorithm 
        
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) # TODO:  en dd usan 0.5*torch.randn(shape, device=device) y no en mtdiff

        chain = [x] if return_chain else None  # TODO: condicionar a s0 si o no? 
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)): # TODO repaint sampling ... 
            t = make_timesteps(batch_size, i, device)
            x = self.p_sample(self, x, t)

            progress.update({'t': i})
            if return_chain: chain.append(x)

        progress.stamp() # en otros usar .close()

        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x,  chain)

    @torch.no_grad()
    def conditional_sample(self, horizon=None,batch_size=32, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        ### opcion 1 ###
        mode_batch=self.bernoulli_dist.sample(sample_shape=(x_noisy.size(0), )).to(x_noisy.device).long() # (B,1) 0 o 1...
       # print("mode",mode_batch)
        mask=self.get_mask_from_batch(mode_batch) # (B,H,T) same dims as x 
       # print("mask0",mask[0,:,:])
      #  print("mask1",mask[1,:,:])
        assert mask.shape==x_noisy.shape # TODO sacar assert 

        x_noisy=x_start*mask+x_noisy*(1-mask) # conditioning with mask... # TODO check this (works and the mask do what we want... )
       # print(x_start[0,:,:],"start0")
      #  print(x_start[1,:,:],"start1")

       # print("xnoisy0",x_noisy[0,:,:])
      #  print("xnoisy1",x_noisy[1,:,:])
        #noise=noise*(1-mask) # Check also this
        loss_weights=self.loss_weights*(1-mask)
     #   print("Noise to pred 0",loss_weights[0,:,:])
    #    print("Noise to pred 1",loss_weights[1,:,:])
        ############

        ### opcion 2 ### faster... duplicate the batch... 
       #  mode=self.bernoulli_dist.sample(sample_shape=(x_noisy.size(0), 1)).to(x_noisy.device).float() # (B,1) 0 o 1...
        
      #  mask=get_mask_from_batch(mode_batch,mask_dict) (B,H,T) same dims as x 
        
    #    x_noisy=x_start*mask+x_noisy*(1-mask) # conditioning with mask... # TODO check this (works and the mask do what we want... )
        #noise=noise*(1-mask) # Check also this
        ############
        
#TODO APLICAR RANDOM CONDITIONING PARA LOS MODOS... 
# TODO ver para pasar multiples task y que dataset tenga bien repartidos los task... en metaworld 
        #mode...
        mode_batch=mode_batch.float().unsqueeze(-1)

        mode_batch.requires_grad = True
        t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        x_noisy.requires_grad= True
        noise.requires_grad = True

        pred_epsilon = self.model(x_noisy,t,mode_batch)

        assert noise.shape == pred_epsilon.shape

        loss = self.loss_fn(pred_epsilon, noise,loss_weights=loss_weights)

        return loss, {}
    

    def loss(self, x): 
    
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, t)



    def forward(self,traj_known,mask, *args, **kwargs):
        return self.conditional_sample(traj_known,mask, *args, **kwargs) # TODO: repaint sampling... faltan hiperparametros del sampling... 
    



    # flavors1: raw, no condition on first state, action and test with added loss weight
    # 2 condition on first state and rtg
    # differiented reward function...
    # conditioned on returns
    # conditioned on mode ( exploration, task inference, planning) or: task planning and exploration