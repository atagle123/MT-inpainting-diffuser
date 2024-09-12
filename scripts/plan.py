import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffuser.sampling.policies import Policy
from diffuser.utils.setup import load_experiment_params,set_seed
import diffuser.utils as utils
import wandb
import imageio.v2 as iio
import numpy as np
import torch
from datetime import datetime
from diffuser.utils.rollouts import TrajectoryBuffer
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dataset="maze2d"

args=load_experiment_params(f"logs/configs/{dataset}/configs_diffusion.txt")

set_seed(args["seed"]) # TODO maybe change this... 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## diffusion model
horizon= args["horizon"]
n_diffusion_steps= args["n_diffusion_steps_sample"]


## loading
current_dir=os.getcwd()

#ponerle nombre al experimento

exp_name="123"#f"H{horizon}_T{n_diffusion_steps}_d{discount}_s{args["scale"]}_b{args["batch_size"]}"

diffusion_loadpath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"diffusion", exp_name)

savepath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"plans", exp_name)

os.makedirs(savepath,exist_ok=True)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    diffusion_loadpath,
    epoch="latest", seed=args["seed"],
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
#dataset.inference_mode()


logger_config = utils.Config(
    utils.Logger,
    logpath=savepath
)

logger = logger_config()

policy_config = utils.Config(
    Policy,
    diffusion_model=diffusion,
    dataset=dataset,
    gamma=args["gamma"],
    ## sampling kwargs
    batch_size_sample=args["batch_size_sample"],
    horizon_sample = args["horizon_sample"],
    return_chain=args["return_chain"]
)

policy = policy_config()


###
#test
###
import torch
import einops
import wandb

from diffuser.utils.arrays import batch_to_device

def cycle(dl):
    while True:
        for data in dl:
            yield data

import torch.nn as nn

# Initialize MSE Loss function
mse_loss_fn = nn.MSELoss()

def measure_task_inference_error():
    dataloader = cycle(torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
            ))

    batch = next(dataloader)
    batch = batch_to_device(batch)  # batch to device # check this... # TODO maybe it can perform quicly
    mode_batch=torch.tensor([1]).to(device)

    traj = policy(*batch, mode_batch)
    pred=torch.mean(traj[:,:,-2:],dim=(0,1))
    target=torch.mean(batch.trajectories[0,:,-2:],dim=0)
    loss = mse_loss_fn(pred, target)
    return(loss)

losses=[]
for i in range(500):
    loss=measure_task_inference_error()
    losses.append(loss)

# Compute the mean of the losses
mean_loss = sum(losses) / len(losses)

print("\nMean Loss:")
print(mean_loss)
  #  print(traj[:,:,:-2],"pred")
   # print(batch.trajectories[0,:,:-2],"actual")
  #  print( (traj[0:,:,:-2]==batch.trajectories[0,:,:-2]).all())
#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#


env = dataset.minari_dataset.recover_environment(render_mode="rgb_array_list") # probar con rgb array list
#env = gym.make("HalfCheetah-v5",render_mode="rgb_array_list")

seed = int(datetime.now().timestamp()) # TODO maybe change this... 
print(f"Using seed:{seed}")

observation, _ = env.reset(seed=seed)   

wandb_log=args["wandb_log"]

if wandb_log:
        
    wandb.init(
        project='Diffusion_RL_thesis',
        name=exp_name,
        monitor_gym=True,
        save_code=True)
    
print(savepath, flush=True)

rollouts=TrajectoryBuffer()

total_reward = 0
for t in range(args["max_episode_length"]):

    ## format current observation for conditioning
    conditions = {0: observation} # TODO change observations... 
    action, samples = policy(conditions, batch_size=args["batch_size"])
    ## execute action in environment
    next_observation, reward, terminated, truncated, _ = env.step(action)
    ## print reward and score
    total_reward += reward

   # clave que el max episode lenght sea el mismo que el con el que se recolecto el dataset, para el score, si no se tienen scores diferentes.
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | '
        f'values: {samples.values} | scale: {args["scale"]}',
        flush=True,)


    rollouts.add_transition(observation, action, reward, next_observation, terminated,total_reward)

    if terminated:
        break
    

    observation = next_observation

rollouts.end_trajectory()
rollouts.save_trajectories(filepath=os.path.join(savepath,f'rollout_{seed}.pkl'))


filename = f'rollout_video_{seed}.mp4'
filepath=os.path.join(savepath,filename)
writer = iio.get_writer(filepath, fps=env.metadata["render_fps"])

frames=env.render()
# Close the environment
env.close()

for frame in frames:
    frame = (frame * -255).astype(np.uint8)
    writer.append_data(frame)
writer.close()

if wandb_log: wandb.log({"video": wandb.Video(filepath)})


## write results to json file at `args.savepath`
logger.finish(t, total_reward, terminated, diffusion_experiment,seed,args["scale"],args["batch_size"])
