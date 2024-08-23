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
RewardBatch = namedtuple('Batch', 'trajectories conditions returns')


class InpaintSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env_name='halfcheetah-expert-v0', horizon=64,
        normalizer="normalization.GaussianNormalizer", preprocess_fns=[], max_path_length=1000, # cambiar normalizer por true o false, ver si en un futuro usamos otros...
        max_n_episodes=10000, termination_penalty=0, seed=None,use_padding=True, include_returns=True, normed_keys=['observations', 'actions','rewards'], p_mask=0.5): 
        
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.termination_penalty=termination_penalty
        self.use_padding=use_padding
        self.include_returns=include_returns
        self.p_mask=p_mask

        self.minari_dataset=minari.load_dataset(env_name)
        self.minari_dataset.set_seed(seed=seed)
        self.env = self.minari_dataset.recover_environment()
        action_space=self.env.action_space
        observation_space = self.env.observation_space
        del self.env #save memory

        assert self.minari_dataset.total_episodes<= max_n_episodes
        # hacer un assert del path lenght

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]

        
        normalizer=import_class(normalizer)
        self.normalizer=normalizer(self.minari_dataset,keys=normed_keys,use_padding=self.use_padding,max_len=self.max_path_length)
        
        self.make_dataset(normed_keys=normed_keys)
        self.make_indices(horizon)
        
    def make_dataset(self,normed_keys=['observations', 'actions']): # add new field total return
        """
        Format: episodes_dict.keys-> ["observations","actions","rewards","terminations","truncations","total_returns"]
                episodes_dict.values-> np.array 2d [H,Dim]  #revisar
        """
        episodes_generator = self.minari_dataset.iterate_episodes()
        episodes_dict={}
        ### generate new dataset in the format ###

        for episode in episodes_generator:
            dict={}
            for key in ["observations","actions","rewards","terminations","truncations"]:
                attribute=getattr(episode,key)
                attribute_2d=atleast_2d(attribute)
                attribute=pad(attribute_2d,max_len=self.max_path_length) #pad pad before than normalizer? and for all the keys

                if key in normed_keys:
                    attribute=self.normalizer.normalize(attribute,key) # normalize

                #attribute=pad(attribute_2d,max_len=self.max_path_length)
                assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1]) # check normalized dims 

                if key=="rewards":  # ojo aca donde poner esto... antes o despues de normalizar... 

                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
                

                    # hacer suma cumulativa para ver total return de la trayectoria... y normalizar ...

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
        return {0: observations[0]} # 50/ 50 mask and uniform mask over the whole horizon...
    
    def inference_mode(self):
        del self.episodes; del self.indices; del self.normalizer.minari_dataset  #save memory


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        ep_id, start, end = self.indices[idx]
        episode=self.episodes[ep_id]  # normed episode # ojo con los id checkear que esten bien... 

        observations = episode['observations'][start:end]
        actions = episode['actions'][start:end]
        rewards=episode['rewards'][start:end]

        mask=np.random.binomial(n=1, p=self.p_mask)
        if mask:
            raise NotImplementedError
            #mask...

        """        conditions = np.ones((self.horizon, 2*traj_dim)).astype(np.float32)

        # Set up conditional masking
        conditions[t_step:,:self.action_dim] = 0
        conditions[:,traj_dim:] = 0
        conditions[t_step,traj_dim:traj_dim+self.action_dim] = 1
        """

        conditions = self.get_conditions(observations) # fix this... is the mask...

        trajectories = np.concatenate([actions, observations,rewards], axis=-1) # check this

        if self.include_returns:
            """
                Para normalizar, primero normalizar rewards (0,1), calcular reward to go de cada estado (cierto gamma), normalizar con formula de gammas... y tenemos el reward to go normalizados de todos los estados (deberian ser similares en treyactorias optimas), luego renormalizar rewards to go para q esten si o si en el rango 0,1 y (condicionar a eso...) despyues al hacer el mask condicionar el rtg desde el estado q se esta midiendo (quizas rtg promedio? o descontado tiene sentido hcaerlo para cada estado en todo caso)... y no desde toda la historia. 
            """
            returns=episode['returns'] # deberia dar solo un valor..., ver si hacer el reward to go quizas, tiene mas sentido... 
            batch = RewardBatch(trajectories, conditions, returns) # probar esto, el contra argumento es que el las rewards pasadas pudieron haber sido buenas, lo q no implica q las futuras sean buenas. quizas condicionar en returns y reward to go... 
        else:
            batch = Batch(trajectories, conditions)

        return batch



    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
     print("I'm being unpickled with these values: " + repr(d))
     self.__dict__ = d


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
       # normed_values = normed_values * 2 - 1

        normed_values = np.array([normed_values], dtype=np.float32)
        return normed_values
    
    def get_linear_mask(self):
        raise NotImplementedError



class GoalDataset(InpaintSequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

