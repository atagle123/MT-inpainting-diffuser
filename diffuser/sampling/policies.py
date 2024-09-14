from collections import namedtuple
import torch
import numpy as np
from diffuser.utils.arrays import atleast_2d
Trajectories = namedtuple('Trajectories', 'actions observations rewards task')


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


def sort_by_values(actions, observations, rewards,gamma): # refactorizar funcion
    """
    [B,H,(A+S+R)]
    """
    values=compute_reward_to_go_batch(rewards,gamma) # (B,H,1)-> (B)

    inds = torch.argsort(values, descending=True)

    actions_sorted = actions[inds]
    observations_sorted=observations[inds]
    rewards_sorted=rewards[inds]
    values = values[inds]

    return actions_sorted,observations_sorted,rewards_sorted, values



class Policy:
    """
    Policy base class
    """

    def __init__(self, diffusion_model, dataset,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs

    def __call__(self):
       raise NotImplementedError

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    
    def norm_evertything(self,trajectory,keys_order): # TODO test... maybe this has to be a method of dataset... check dims bewtween numpy and torch if it can generalize... 
        previous_dim=0
        for key in keys_order: # TODO construct dims one time only...
            current_dim=self.dataset.keys_dim_dict[key]+previous_dim
            unnormed_attribute = trajectory[:, :, previous_dim:current_dim]
            trajectory[:, :, previous_dim:current_dim]=self.dataset.normalizer.normalize(unnormed_attribute, key)
            previous_dim=current_dim
        return(trajectory)
    
    def unorm_everything(self,trajectory,keys_order): # TODO test...
        previous_dim=0
        for key in keys_order: # TODO construct dims one time only...
            current_dim=self.dataset.keys_dim_dict[key]+previous_dim
            unnormed_attribute = trajectory[:, :, previous_dim:current_dim]
            trajectory[:, :, previous_dim:current_dim]=self.dataset.normalizer.normalize(unnormed_attribute, key) 
            previous_dim=current_dim
        return(trajectory)

# idea: policy tiene metodos basicos y el metodo call depende del caso, por ejemplo en el exp rtg se llama a task inference con la historia ya conocida y tambien se llama a la eleccion de acciones... 


class Policy_mode(Policy): # TODO falta super init

    def __init__(self, diffusion_model, dataset,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs
        self.keys_order=("actions", "observations","rewards","returns","task")

    def __call__(self, rollouts):
        """
        Main policy function that normalizes the data, calls the model and returns the result

        Args:
            rollouts (Class): class that has trajectories history... 
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """

        # 2. constructs the mask or the mode depending the case
        # batchifies the mode and the known batch
        # pass to the model
        # obtain the results
        # unnormalize, sort by rewards? and then pick the first action...
        # 

        conditions=self.get_last_traj_rollout_torch(rollouts)
        normed_conditions=self.norm_evertything(conditions,keys_order=self.keys_order)
        
        # here batch the normed conditions and see what to do... 

        ## run reverse diffusion process
        trajectories = self.diffusion_model(traj_known=conditions, mode=mode, **self.sample_kwargs) # 

        trajectories=trajectories.trajectories
     #   print(trajectories[:, :, -self.task_dim:])
        #normed_rewards = trajectories[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        #rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        #normed_actions = trajectories[:, :, :self.action_dim]
        #actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

      #  normed_observations = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim]
      #  observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observation')

      #  normed_task = trajectories[:, :, -self.task_dim:] # TODO check... maybe do one function per sampling mode... 
       # task = self.dataset.normalizer.unnormalize_torch(normed_task, 'desired_goal')

        #TODO ver donde pponer el task sorted.. 
       # actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 
        # TODO ver que es mejor, si ordenar el rtg o los rewards... 
        ## extract first action
     #   action = actions_sorted[0, 0]

      #  trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted,task) # generalizar aca ...

      #  return action, trajectories,task
        return(trajectories)



    def get_last_traj_rollout_torch(self,rollouts): # TODO maybe win time using the same policy class to not construct the torch rollout every time... 
        states_array,actions_array,rewards_array,total_reward_array,dones_array=rollouts.rollouts_to_numpy(index=-1)
        atleast_2d(states_array) # use a named tuple # TODO do the atleast2d... 
        states_array=states_array[:-1,:] # H,state_dim
        unkown_part=np.zeros(states_array.shape[0],self.task_dim+1) # corresponds to the task and the reward to go...
        known_trajectory=np.concatenate([actions_array, states_array,rewards_array,unkown_part], axis=-1) # TODO this is the specific order... 

        known_trajectory_torch=torch.from_numpy(known_trajectory)
        return(known_trajectory_torch)





class Policy_repaint(Policy):

    def __init__(self, diffusion_model, dataset,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs
        self.keys_order=("actions", "observations","rewards","returns","task")

    def __call__(self, rollouts):
        """
        Main policy function that normalizes the data, calls the model and returns the result

        Args:
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """
        conditions=self.get_last_traj_rollout_torch(rollouts)
        normed_conditions=self.norm_evertything(conditions,keys_order=self.keys_order)

        # batchify masks and traj known... 
        trajectories = self.diffusion_model(traj_known=conditions, mode=mode, **self.sample_kwargs) # 

        trajectories=trajectories.trajectories
     #   print(trajectories[:, :, -self.task_dim:])
        #normed_rewards = trajectories[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        #rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        #normed_actions = trajectories[:, :, :self.action_dim]
        #actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

      #  normed_observations = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim]
      #  observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observation')

      #  normed_task = trajectories[:, :, -self.task_dim:] # TODO check... maybe do one function per sampling mode... 
       # task = self.dataset.normalizer.unnormalize_torch(normed_task, 'desired_goal')

        #TODO ver donde pponer el task sorted.. 
       # actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 
        # TODO ver que es mejor, si ordenar el rtg o los rewards... 
        ## extract first action
     #   action = actions_sorted[0, 0]

      #  trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted,task) # generalizar aca ...

      #  return action, trajectories,task
        return(trajectories) 

    def get_last_traj_rollout_torch(self,rollouts): # TODO maybe win time using the same policy class to not construct the torch rollout every time... 
        states_array,actions_array,rewards_array,total_reward_array,dones_array=rollouts.rollouts_to_numpy(index=-1)
        atleast_2d(states_array) # use a named tuple # TODO do the atleast2d... 
        states_array=states_array[:-1,:] # H,state_dim
        unkown_part=np.zeros(states_array.shape[0],self.task_dim+1) # corresponds to the task and the reward to go...
        known_trajectory=np.concatenate([actions_array, states_array,rewards_array,unkown_part], axis=-1) # TODO this is the specific order... 

        known_trajectory_torch=torch.from_numpy(known_trajectory)
        return(known_trajectory_torch)
    

    def get_masks(): # TODO view a way to build the mask... possible masks: action inference, task inference, both together, previous + moving index of history... 
        pass 