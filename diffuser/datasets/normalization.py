import numpy as np
import torch
from diffuser.utils.arrays import atleast_2d
#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    def __init__(self,minari_dataset,keys,use_padding,max_len):
        self.minari_dataset=minari_dataset
        self.use_padding=use_padding
        self.max_len=max_len
        self.make_params(keys=keys)

    def make_params(self,keys=['observations', 'actions']):

        dict_of_list = {key: [] for key in keys} # cambiar nombre
        
        episodes_generator = self.minari_dataset.iterate_episodes()

        ### Calculate means and std and normalize dataset ###
        for episode in episodes_generator:
            for key in keys:
                attribute=getattr(episode,key)
                attribute=atleast_2d(attribute)

               # if self.use_padding:
    
                  #  attribute=pad(attribute,max_len=self.max_len)

                dict_of_list[key].append(attribute)

        params_dict={}
        for key in keys:
            dict_of_list[key]=np.concatenate(dict_of_list[key]) # all episodes concatenated
            #dict_of_list[key]=dict_of_list[key].astype(np.float32)
            mean=dict_of_list[key].mean(axis=0)
            std=dict_of_list[key].std(axis=0)  #ojo con el dtype
            min=dict_of_list[key].min(axis=0)
            max=dict_of_list[key].max(axis=0)
            params_dict[key]={"mean":mean,"std":std,"min":min,"max":max}

        self.params_dict=params_dict
    
    def __call__(self, x, key):
        return self.normalize(x, key)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()
    

    


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, minari_dataset,keys,use_padding,max_len):
        super().__init__(minari_dataset,keys,use_padding,max_len)
        self.field_normalizer={}

    def normalize(self,unnormed_data, key):
        
        self.field_normalizer[key]="gaussian"

        normed_data=(unnormed_data-self.params_dict[key]["mean"])/self.params_dict[key]["std"]

        return(normed_data)


    def unnormalize(self,normed_data,key):

        unnormed_data=normed_data*self.params_dict[key]["std"]+self.params_dict[key]["mean"]

        return(unnormed_data)
    
    def normalize_torch(self,unnormed_data,key,device="cuda:0"):
        #self.field_normalizer[key]="gaussian"
        torch_mean=torch.from_numpy(self.params_dict[key]["mean"]).to(unnormed_data.device)
        torch_std=torch.from_numpy(self.params_dict[key]["std"]).to(unnormed_data.device)
        
        assert torch_mean.shape==torch_std.shape
        assert torch_mean.shape==unnormed_data[0,0,:].shape

        normed_data=(unnormed_data-torch_mean)/torch_std

        return(normed_data)
    
    def unnormalize_torch(self,normed_data,key,device="cuda:0"):
        torch_mean=torch.from_numpy(self.params_dict[key]["mean"]).to(normed_data.device)
        torch_std=torch.from_numpy(self.params_dict[key]["std"]).to(normed_data.device)

        assert torch_mean.shape==torch_std.shape
        assert torch_mean.shape==normed_data[0,0,:].shape

        unnormed_data=normed_data*torch_std+torch_mean
        return(unnormed_data)

    
    def get_field_normalizers(self):
        return(self.field_normalizer)