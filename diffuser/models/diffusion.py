from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
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
        action_weight=1.0, loss_discount=1.0):
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

    def get_random_mask(self):
        #TODO samplear de una uniforme y mitad mitad... o quizas menos...
        raise NotImplementedError

 
    def get_mask_loss_weights(self,K):
        '''
            sets loss coefficients for masked trajectory
            
            K: The masked past step
        TODO CAMBIAR ACA EL MASK... ver como hacer para los batches... 
        '''

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        #discounts = self.loss_discount ** torch.arange(self.horizon-K, dtype=torch.float) #TODO: horizon es todo el horizonte y h-K es lo que hay q predecir... 
        #discounts = discounts / discounts.mean() # TODO: para usar los discount podemos calcularlos antes... y sumarlos al mask 

        #loss_weights = torch.einsum('h-k,t->(h-k)t', discounts, dim_weights)

        loss_weights[K, :self.action_dim] = self.action_weight
        loss_weights[K, self.action_dim:self.action_dim+self.observation_dim] = 0 # because the initial state is given 

        return loss_weights



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

        pred_epsilon = self.model(x_noisy, returns, t,use_dropout=True) # TODO: maybe add task and actual K step. ver modelo tambien

        assert noise.shape == pred_epsilon.shape

        loss, info = self.loss_fn(pred_epsilon, noise) # TODO: falta ver la funcion de loss... 

        return loss, info


    def loss(self, x, returns, *args): # maybe add task and actual K step.

        batch_size = len(x) # TODO aca mtdiff hace algo con einops...
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        K_step_mask = torch.randint(0, self.horizon, (batch_size,), device=x.device).long() # TODO: quizas horizon-1 ... usar tambien lo de con cierta prob no usar el masking... quizas multiplicar por una matriz random de 0 y 1... 
        # quizas para el mask duplicar el batch?

        return self.p_losses(x, returns, K_step_mask, t) # TODO: maybe add task and actual K step.



    def forward(self, past_K_history, *args, **kwargs):
        return self.conditional_sample(past_K_history, *args, **kwargs) # TODO: repaint sampling... 

