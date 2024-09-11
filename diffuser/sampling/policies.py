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
    values=compute_reward_to_go_batch(rewards,gamma) # (B,H,1)-> (B)

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
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, mode, verbose=True):
        """
        Function to to get the data from a dict of args

        Args:
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """
        normed_rewards_condition = conditions[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        conditions[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1] = self.dataset.normalizer.normalize_torch(normed_rewards_condition, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        normed_actions_condition = conditions[:, :, :self.action_dim]
        conditions[:, :, :self.action_dim] = self.dataset.normalizer.normalize_torch(normed_actions_condition, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

        normed_observations_condition = conditions[:, :, self.action_dim:self.action_dim+self.observation_dim]
        conditions[:, :, self.action_dim:self.action_dim+self.observation_dim] = self.dataset.normalizer.normalize_torch(normed_observations_condition, 'observations')

        normed_task_condition = conditions[:, :, -self.task_dim:] # TODO check... 
        conditions[:, :, -self.task_dim:] = self.dataset.normalizer.normalize_torch(normed_task_condition, 'desired_goal')
        

        ## run reverse diffusion process
        trajectories = self.diffusion_model(conditions, mode, verbose=verbose, **self.sample_kwargs) # 

        normed_rewards = trajectories[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        normed_actions = trajectories[:, :, :self.action_dim]
        actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

        normed_observations = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim]
        observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observations')

        normed_task = trajectories[:, :, -self.task_dim:] # TODO check... maybe do one function per sampling mode... 
        task = self.dataset.normalizer.normalize_torch(normed_task, 'desired_goal')

        #TODO ver donde pponer el task sorted.. 
        actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 
        # TODO ver que es mejor, si ordenar el rtg o los rewards... 
        ## extract first action
        action = actions_sorted[0, 0]

        trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted,task) # generalizar aca ...

        return action, trajectories, values

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    
    def norm_evertything(self,):
        pass

    def unorm_everything(self,):
        pass

