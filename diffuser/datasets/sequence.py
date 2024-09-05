from collections import namedtuple
import numpy as np
import torch
import gymnasium as gym
from .normalization import GaussianNormalizer,LimitsNormalizer
from .preprocessing import get_preprocess_fn
import minari
from diffuser.utils.config import import_class
from diffuser.utils.arrays import atleast_2d,pad, pad_min

Batch = namedtuple('Batch', 'trajectories')
#RewardBatch = namedtuple('Batch', 'trajectories returns')

#TaskRewardBatch=namedtuple('Batch', 'trajectories returns task')


def find_key_in_data(data, key_to_find):
    def recursive_search(item):
        # If item is a dictionary
        if isinstance(item, dict):
            if key_to_find in item:
                result = item[key_to_find]
                if isinstance(result, np.ndarray):
                    return result
                # Continue searching in values if the result is not found
            for value in item.values():
                result = recursive_search(value)
                if result is not None:
                    return result
        
        # If item is a list
        elif isinstance(item, list):
            for element in item:
                result = recursive_search(element)
                if result is not None:
                    return result
        
        # If item is a NumPy array (though arrays don't have keys, skip them)
        elif isinstance(item, np.ndarray):
            return None
        
        # If item has attributes
        elif hasattr(item, '__dict__'):
            if key_to_find in item.__dict__:
                result = getattr(item, key_to_find)
                if isinstance(result, np.ndarray):
                    return result
            for value in item.__dict__.values():
                result = recursive_search(value)
                if result is not None:
                    return result
        
        # If item has a method named `key_to_find`
        elif hasattr(item, key_to_find):
            result = getattr(item, key_to_find)
            if isinstance(result, np.ndarray):
                return result
        
        return None
    
    return recursive_search(data)


class InpaintSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name='halfcheetah-expert-v0', horizon=64,
        normalizer="datasets.normalization.GaussianNormalizer", max_path_length=1000, # cambiar normalizer por true o false, ver si en un futuro usamos otros...
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

        self.make_returns_fast()

        self.normalize_dataset(normed_keys=self.normed_keys,normalizer=import_class(normalizer))

        self.sanity_test() # TODO 
        
    def make_dataset(self,view_keys):
        """
        Transforms minari dataset to a standard way... 

        Format: episodes_dict.keys-> ["observations","actions","rewards","terminations","truncations","total_returns"]
                episodes_dict.values-> np.array 2d [H,Dim]
        """ 
        print("Making dataset... ")
        episodes_generator = self.minari_dataset.iterate_episodes()
        self.episodes={}
        ### generate new dataset in the format ###
        # TODO agregar un truncador de observaciones si es que son una dim mas que las acciones... considerar tambien el caso del maze2d 
        # TODO maybe change name of observation to observations... 
        for episode in episodes_generator:
            dict={}
            for key in view_keys:
                
                attribute=find_key_in_data(episode,key)

                if attribute is None:
                    raise KeyError(f" Couldn't find a np.array value for the key {key}")

                attribute_2d=atleast_2d(attribute)

                if self.use_padding:
                    attribute=pad(attribute_2d,max_len=self.max_path_length)

                    assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1])

                else:
                    attribute=pad_min(attribute_2d,min_len=self.horizon)

                if key=="rewards":  
                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
                        
                
                dict[key]=attribute

            self.episodes[episode.id]=dict

    def normalize_dataset(self,normed_keys,normalizer):
        print("Normalizing dataset... ")

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
        print("Making indices... ")
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
        del self.episodes; del self.indices #save memory in inference 


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
        print("Making returns... ")

        discount_array=self.discount ** np.arange(self.max_path_length) # (H)
        discount_array=atleast_2d(discount_array) # (H,1)
        for ep_id, dict in self.episodes.items():
           # print(f"calc_return {ep_id}")
            rtg=[]
            rewards=dict["rewards"]
            for i in range(len(rewards)):
                future_rewards=rewards[i:]
                horizon=len(future_rewards)
                norm_rtg=self.calculate_norm_rtg(future_rewards, horizon=horizon, discount=self.discount, discount_array=discount_array[:horizon]) # es enecesario normalizar rtg 
                rtg.append(np.exp(norm_rtg)) # TODO CHeck this
            
            returns_array=np.array(rtg,dtype=np.float32)
            assert returns_array.shape[0]==rewards.shape[0]

            self.episodes[ep_id]["returns"]=atleast_2d(returns_array)

        self.normed_keys.append("returns")

    def make_returns_fast(self): # TODO idea gamma deberia ser el mismo con el que se testea ... ✓
        print("Making returns fast... ")
        
        discount_array=self.discount ** np.arange(self.max_path_length) # (H)
        discount_array=atleast_2d(discount_array)
        norm_factors=[]
        for horizon in range(self.max_path_length):
            norm_factors.append(self.calc_norm_factor(self.discount,horizon)) # list with list[horizon]-> norm_factor(horizon)

        for ep_id, dict in self.episodes.items():
           # print(f"calc_return {ep_id}")
            rtg_list=[]
            rewards=dict["rewards"]
            horizon=len(rewards)
           # print(rewards*discount_array[:horizon])
            rtg_partial=np.sum(rewards*discount_array[:horizon]) # (H)*(H)-> 1 #TODO check this
            #print(rtg_partial*norm_factors[horizon],horizon,rewards,norm_factors[horizon])
            rtg_list.append(np.exp(rtg_partial*norm_factors[horizon]))
            for rew in rewards:
                rtg_partial=(rtg_partial-rew[0])/self.discount
               # print(rtg_list,rtg_partial)
                horizon-=1
               # print("hor",horizon)
                rtg_list.append(np.exp(rtg_partial*norm_factors[horizon])) # TODO CHeck this
            
            returns_array=np.array(rtg_list[:-1],dtype=np.float32)
            assert returns_array.shape[0]==rewards.shape[0]

            self.episodes[ep_id]["returns"]=atleast_2d(returns_array)

        self.normed_keys.append("returns")

    
    def calc_norm_factor(self,discount,horizon):
        norm_factor=(1-discount)/(1-discount**(horizon+1))
        return(norm_factor)


    def calculate_norm_rtg(self,rewards,horizon,discount,discount_array):

            rtg=np.sum(rewards*discount_array) # (H,1)*(H,1)-> 1 #TODO check this

            norm_factor=(1-discount)/(1-discount**(horizon+1))
            norm_rtg=rtg*norm_factor
            return(norm_rtg)

    def sanity_test(self):
        raise  NotImplementedError
        #check dims... 

class Maze2d_inpaint_dataset(InpaintSequenceDataset):
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
        observation_space = self.env.observation_space["observation"]

        assert self.minari_dataset.total_episodes<= max_n_episodes

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]
        
        self.make_dataset(view_keys=view_keys)
        self.make_indices(horizon)

        self.make_returns_fast()

        self.normalize_dataset(normed_keys=self.normed_keys,normalizer=import_class(normalizer))

      #  self.sanity_test() # TODO 

    def __getitem__(self, idx):
        ep_id, start, end = self.indices[idx] 
        episode=self.episodes[ep_id]  

        observations = episode['observation'][start:end] # TODO make this generalizable using view_keys ore something like that... 
        actions = episode['actions'][start:end]
        rewards=episode['rewards'][start:end]
        returns=episode["returns"][start:end]
        task=episode["desired_goal"][start:end]

        trajectories = np.concatenate([actions, observations,rewards,task,returns], axis=-1) # check this

        batch = Batch(trajectories)

        return batch


