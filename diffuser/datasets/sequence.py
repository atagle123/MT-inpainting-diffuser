from collections import namedtuple
import numpy as np
import torch
import gymnasium as gym
from .normalization import GaussianNormalizer
from .preprocessing import get_preprocess_fn
import minari
from diffuser.utils.config import import_class
from diffuser.utils.arrays import atleast_2d,pad

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env_name='halfcheetah-expert-v0', horizon=64,
        normalizer="normalization.GaussianNormalizer", preprocess_fns=[], max_path_length=1000, # cambiar normalizer por true o false, ver si en un futuro usamos otros...
        max_n_episodes=10000, termination_penalty=0, seed=None,normed=True,use_padding=True): 
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.termination_penalty=termination_penalty
        self.use_padding=use_padding

        self.minari_dataset=minari.load_dataset(env_name)
        self.minari_dataset.set_seed(seed=seed)
        self.env = self.minari_dataset.recover_environment()
        action_space=self.env.action_space
        observation_space = self.env.observation_space
        self.horizon=horizon
        del self.env #save memory

        assert self.minari_dataset.total_episodes<= max_n_episodes
        # hacer un assert del path lenght

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]

        self.normed=normed
        
        normalizer=import_class(normalizer)
        self.normalizer=normalizer(self.minari_dataset,keys=['observations', 'actions'],use_padding=self.use_padding,max_len=self.max_path_length)
        
        self.make_dataset()
        self.make_indices(horizon)
        
    def make_dataset(self,normed_keys=['observations', 'actions']):

        episodes_generator = self.minari_dataset.iterate_episodes()
        episodes_dict={}
        ### generate new dataset in the format ###

        for episode in episodes_generator:
            dict={}
            for key in ["observations","actions","rewards","terminations","truncations"]:
                attribute=getattr(episode,key)
                attribute_2d=atleast_2d(attribute)
                attribute=pad(attribute_2d,max_len=self.max_path_length) #pad pad before than normalizer? and for all the keys

                if key in normed_keys and self.normed:
                    attribute=self.normalizer.normalize(attribute,key) # normalize

                #attribute=pad(attribute_2d,max_len=self.max_path_length)
                assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1]) # check normalized dims 

                if key=="rewards":
                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
    
                dict[key]=attribute
            episodes_dict[episode.id]=dict

        self.episodes=episodes_dict

    def make_indices(self, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []

        episodes_generator = self.minari_dataset.iterate_episodes() # quizas aca usar el dict, aunque si esta bien implementado no deberian haber errores
        
        for episode in episodes_generator:  # assumes padding fix it to use no padding
            
            episode_lenght=episode.total_timesteps
            assert self.max_path_length>=episode_lenght
            

            max_start = min(episode_lenght - 1, self.max_path_length - horizon)

            if not self.use_padding:
   
                max_start = min(episode_lenght - horizon,self.max_path_length - horizon) # if episode lenght<horizon max start will be negative, doesnt have much sense use padding when episodes are truncated
                
                assert episode_lenght>=horizon

            for start in range(max_start+1):
                end = start + horizon  
                indices.append((episode.id, start, end))

        indices = np.array(indices)
        self.indices=indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    
    def inference_mode(self):
        del self.episodes; del self.indices; del self.normalizer.minari_dataset  #save memory


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        ep_id, start, end = self.indices[idx]
        episode=self.episodes[ep_id]  # normed episode # ojo con los id checkear que esten bien... 

        observations = episode['observations'][start:end]
        actions = episode['actions'][start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
     print("I'm being unpickled with these values: " + repr(d))
     self.__dict__ = d


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):  # ojo que normaliza los valores, no las rewards, es sobre las trayctorias
        if not self.normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True
        ## [0, 1]
        normed_values = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed_values = normed_values * 2 - 1
        return normed_values

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        ep_id, start, end = self.indices[idx]

        episode=self.episodes[ep_id]
        rewards = episode['rewards'][start:end]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()

        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
    
    def unormalize_value(self,value):
        raise NotImplementedError()
    
    def inference_mode(self):
        if not self.normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

        del self.episodes; del self.indices; del self.normalizer.minari_dataset  #save memory