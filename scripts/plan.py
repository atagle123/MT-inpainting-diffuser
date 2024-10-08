import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffuser.sampling.policies import Policy_mode_returns_conditioned
from diffuser.utils.setup import load_experiment_params,set_seed
import diffuser.utils as utils
import imageio.v2 as iio
import numpy as np
import torch
from datetime import datetime
from diffuser.utils.rollouts import TrajectoryBuffer
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dataset="maze2d"
exp_name="conditioned_mode_sampling"

args=load_experiment_params(f"logs/configs/{dataset}/{exp_name}/configs_diffusion.txt")

set_seed(args["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## loading
current_dir=os.getcwd()

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
    Policy_mode_returns_conditioned,
    diffusion_model=diffusion,
    dataset=dataset,
    keys_order=args["keys_order"],
    gamma=args["gamma"],
    ## sampling kwargs
    batch_size_sample=args["batch_size_sample"],
    horizon_sample = args["horizon_sample"],
    return_chain=args["return_chain"]
)

policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
    
print(savepath, flush=True)

seed = int(datetime.now().timestamp()) # TODO maybe change this... 
print(f"Using seed:{seed}")

env = dataset.minari_dataset.recover_environment(render_mode="rgb_array_list")
observation, info = env.reset(seed=seed)  

rollouts=TrajectoryBuffer(observation["observation"],info,action_dim=dataset.action_dim)

total_reward = 0
for t in range(args["max_episode_length"]):

    action, samples = policy(rollouts,provide_task=observation["desired_goal"]) 
    ## execute action in environment
    observation, reward, terminated, truncated, info = env.step(action)
    ## print reward
    total_reward += reward

   # clave que el max episode lenght sea el mismo que el con el que se recolecto el dataset, para el score, si no se tienen scores diferentes.
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | ',
        flush=True)

    rollouts.add_transition(observation["observation"], action, reward, terminated,total_reward,info) # TODO this is for maze 

    if terminated or info["success"]==True: # TODO this is for maze 
        break
    
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

## write results to json file at `args.savepath`
logger.finish(t, total_reward, terminated, diffusion_experiment,seed,args["batch_size_sample"])