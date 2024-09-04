from collections import namedtuple
import numpy as np
import torch
import gymnasium as gym
from .normalization import GaussianNormalizer,LimitsNormalizer
from .preprocessing import get_preprocess_fn
import minari
from diffuser.utils.config import import_class
from diffuser.utils.arrays import atleast_2d,pad

Batch = namedtuple('Batch', 'trajectories')
RewardBatch = namedtuple('Batch', 'trajectories returns')

TaskRewardBatch=namedtuple('Batch', 'trajectories returns task')


def find_key_in_data(data, key_to_find):

    result=None
    def recursive_search(item):
        if isinstance(item, dict):
            if key_to_find in item:
                result=item[key_to_find]

                if isinstance(result, np.ndarray):
                    return(result)

            for value in item.values():
                recursive_search(value)

        if isinstance(item, np.ndarray):
            # NumPy arrays do not contain keys, so we skip them
            pass

        if hasattr(item, key_to_find):
            result=getattr(item,key_to_find)
            if isinstance(result, np.ndarray):
                return(result)
            
        if hasattr(item, "__dict__"):
            if key_to_find in item.__dict__.keys():
                result=getattr(item,key_to_find)
                print(result)
                if isinstance(result, np.ndarray):
                    return(result)

            # Handle objects with __dict__ attribute (e.g., class instances)
            for value in item.__dict__.values():
                recursive_search(value)
    
    result=recursive_search(data)
    return result


class InpaintSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name='halfcheetah-expert-v0', horizon=64,
        normalizer="normalization.GaussianNormalizer", max_path_length=1000, # cambiar normalizer por true o false, ver si en un futuro usamos otros...
        max_n_episodes=100000, termination_penalty=0, seed=None,use_padding=True,view_keys=['observations', 'actions','rewards'], normed_keys=['observations', 'actions','rewards'],discount=0.99): 
        
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.termination_penalty=termination_penalty
        self.use_padding=use_padding
        self.discount=discount
        self.normed_keys=normed_keys

        self.minari_dataset=minari.load_dataset(dataset_name)
        self.minari_dataset.set_seed(seed=seed)
        self.env = self.minari_dataset.recover_environment()
        action_space=self.env.action_space
        observation_space = self.env.observation_space

        assert self.minari_dataset.total_episodes<= max_n_episodes

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]
        
        self.make_dataset(view_keys=view_keys)
        self.make_indices(horizon)

        self.make_returns()

        self.normalize_dataset(normed_keys=self.normed_keys,normalizer=import_class(normalizer))

        self.sanity_test() # TODO 
        
    def make_dataset(self,view_keys):
        """
        Transforms minari dataset to a standard way... 

        Format: episodes_dict.keys-> ["observations","actions","rewards","terminations","truncations","total_returns"]
                episodes_dict.values-> np.array 2d [H,Dim]  #revisar  TODO add RTG as field and then normalize across the timestep. and tasks.  
        """ 
        episodes_generator = self.minari_dataset.iterate_episodes()
        self.episodes={}
        ### generate new dataset in the format ###
        # TODO agregar un truncador de observaciones si es que son una dim mas que las acciones... considerar tambien el caso del maze2d 
        # TODO maybe change name of observation to observations... 
        for episode in episodes_generator:
            dict={}
            for key in view_keys:
                try:
                    attribute=find_key_in_data(episode,key)

                except KeyError as e:
                    print(f"{e} Couldn't find a np.array value for the key {key}")

                attribute_2d=atleast_2d(attribute)
                attribute=pad(attribute_2d,max_len=self.max_path_length) 

                assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1]) # check normalized dims 

                if key=="rewards":  
                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
                        
                
                dict[key]=attribute

            self.episodes[episode.id]=dict

    def normalize_dataset(self,normed_keys,normalizer):
        self.normalizer=normalizer(dataset=self.episodes,normed_keys=normed_keys,use_padding=self.use_padding,max_len=self.max_path_length)

        for ep_id, dict in self.episodes.items():
            for key,attribute in dict.items():

                if key in normed_keys:
                    attribute=self.normalizer.normalize(attribute,key) # normalize

                    dict[key]=attribute

            self.episodes[ep_id]=dict # TODO check this... 

    def make_indices(self, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        
        for ep_id,episode_dict in self.episodes.items():  # assumes padding fix it to use no padding
            
            episode_lenght=len(episode_dict["actions"])   # uses actions as reference TODO check this... 

            assert self.max_path_length>=episode_lenght
            
            max_start = min(episode_lenght - 1, self.max_path_length - horizon)

            if not self.use_padding:
   
                max_start = min(episode_lenght - horizon,self.max_path_length - horizon) # if episode lenght<horizon max start will be negative, doesnt have much sense use padding when episodes are truncated
                
                assert episode_lenght>=horizon

            for start in range(max_start+1):
                end = start + horizon
                indices.append((ep_id, start, end))

        indices = np.array(indices)
        self.indices=indices

    
    def inference_mode(self):
        del self.episodes; del self.indices; del self.normalizer.minari_dataset  #save memory TODO check this... 


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_id, start, end = self.indices[idx] 
        episode=self.episodes[ep_id]  

        observations = episode['observations'][start:end] # TODO make this generalizable using view_keys ore something like that... 
        actions = episode['actions'][start:end]
        rewards=episode['rewards'][start:end]
        returns=episode["returns"][start:end]
        task=episode["desired_goal"][start:end]

        trajectories = np.concatenate([actions, observations,rewards,task,returns], axis=-1) # check this

        batch = Batch(trajectories)

        return batch


    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
     print("I'm being unpickled with these values: " + repr(d))
     self.__dict__ = d
    

    def make_returns(self): # TODO idea gamma deberia ser el mismo con el que se testea ... 
        #discount_array=self.discount ** np.arange(self.max_path_length) # (H)
        for ep_id, dict in self.episodes.items():
            rtg=[]
            rewards=dict["rewards"]
            for i in range(len(rewards)):
                future_rewards=rewards[i:]
                norm_rtg=self.calculate_norm_rtg(future_rewards,discount=self.discount) # es enecesario normalizar rtg 
                rtg.append(np.exp(norm_rtg)) # TODO CHeck this
            
            returns_array=np.array(rtg,dtype=np.float32)

            assert returns_array.shape==rewards.shape

            self.episodes[ep_id]["returns"]=returns_array

        self.normed_keys.append("returns")

    
            

    def calculate_norm_rtg(self,rewards,discount):
            horizon=len(rewards)
            discount_array=discount ** np.arange(horizon) # (H)

            discount_rew=rewards*discount_array # (H)*(H)-> H #TODO check this

            rtg=discount_rew.sum()

            norm_factor=(1-discount)/(1-discount**(horizon+1))
            norm_rtg=rtg*norm_factor
            return(norm_rtg)



#class MAze2d_inpaint_dataset(InpaintSequenceDataset):


