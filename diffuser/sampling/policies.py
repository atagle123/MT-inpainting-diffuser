from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn  # que es esta wea


Trajectories = namedtuple('Trajectories', 'actions observations rewards')


def compute_reward_to_go_batch(rewards_batch, gamma):
    """
    Compute the reward-to-go for a batch of reward sequences with a discount factor gamma.
    
    Parameters:
        rewards_batch (torch.Tensor): A 2D tensor of shape (B, Horizon,1) where B is the batch size and Horizon is the length of each sequence.
        gamma (float): The discount factor.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (B) containing reward-to-go values for each sequence in the batch.
    """

    assert rewards_batch.shape[2]==1

    rewards_batch=rewards_batch.squeeze(-1) # (B,H,1) -> (B,H)
    B, H = rewards_batch.shape

    gamma_tensor = torch.pow(gamma, torch.arange(H, dtype=torch.float32)).to(rewards_batch.device)
    gamma_matrix = gamma_tensor.unsqueeze(0).repeat(B, 1) # (B,H)

    # Apply gamma matrix to compute reward-to-go
    reward_to_go_batch = torch.sum(rewards_batch * gamma_matrix, dim=1)  # (B, H) -> (B)
    
    return reward_to_go_batch


def sort_by_values(actions, observations, rewards, no_inpaint_step,gamma): # refactorizar funcion
    """
    [B,past_H+future_H,(A+S+R)]
    """
    rewards_to_go=rewards[:,:no_inpaint_step,:]  # notar que aca no son values (B,future_H,1)

    values=compute_reward_to_go_batch(rewards_to_go,gamma) # (B,H,1)-> (B)

    # multiplicar por gamma 
    # sumar

    inds = torch.argsort(values, descending=True)

    actions_sorted = actions[inds]
    observations_sorted=observations[inds]
    rewards_sorted=rewards[inds]
    values = values[inds]

    return actions_sorted,observations_sorted,rewards_sorted, values



class Policy:

    def __init__(self, diffusion_model, dataset, preprocess_fns,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        """
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """
        
        ## falta
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)   # conditions ahora es todo lo que ya ocurrio... 

        no_inpaint_step=conditions.shape #ver esto.
        #falta

        
        ## run reverse diffusion process
        trajectories = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs) # inpainting... agregar algo para sample by inpainting

        normed_rewards = trajectories[:, :, -1]
        rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        normed_actions = trajectories[:, :, :self.action_dim]
        actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

        normed_observations = trajectories[:, :, self.action_dim:-1] 
        observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observations')

        #IMPIORTANTE: CORTAR ACA POR no inpaint step ...
        actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, no_inpaint_step, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 

        ## extract first action
        action = actions_sorted[0, 0]

        trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted) # generalizar aca ...

        return action, trajectories, values

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
