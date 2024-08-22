from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn  # que es esta wea


Trajectories = namedtuple('Trajectories', 'actions observations values')

def sort_by_values(trajectories): # refactorizar funcion
    """
    [B,H,(A+S+R)]
    """
    rewards=trajectories[:,:,-1] # (B,H,1)

    # multiplicar por gamma 
    # sumar

    inds = torch.argsort(values, descending=True)
    trajectories = trajectories[inds]
    values = values[inds]
    return trajectories, values

class Policy:

    def __init__(self, diffusion_model, dataset, preprocess_fns, **sample_kwargs):
        #self.guide = guide
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        
        ## falta
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)   # conditions ahora es todo lo que ya ocurrio... 

        no_inpaint_step=conditions.shape #ver esto.
        #falta
        ## run reverse diffusion process
        trajectories = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs) # inpainting... agregar algo para sample by inpainting


        trajectories , values = sort_by_values(trajectories) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 


        #trajectories = utils.to_np(trajectories)

        rewards = trajectories[:, :, -1]
        rewards = self.dataset.normalizer.unnormalize_torch(rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        # hacer todo esto en torch

        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        actions = trajectories[:, :, :self.action_dim]
        actions = self.dataset.normalizer.unnormalize_torch(actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:-1] 
        observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observations')
        trajectories = Trajectories(actions, observations, values) # generalizar aca ...
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        """Function that returns a dict with the corresponding tensor"""
        conditions = utils.apply_dict(
            self.dataset.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
