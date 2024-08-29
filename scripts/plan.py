import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffuser.sampling.guides import ValueGuide
from diffuser.sampling.policies import GuidedPolicy
import diffuser.sampling as sampling
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

dataset="halfcheetah"
args=load_experiment_params(f"logs/configs/{dataset}/{dataset}-medium-v2/configs_plan.txt")

set_seed(args["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## diffusion model
horizon= args["horizon"]
n_diffusion_steps= args["n_diffusion_steps"]

## value function
discount= args["discount"]

## loading
diffusion_loadpath= f'diffusion/H{horizon}_T{n_diffusion_steps}'
value_loadpath= f'values/H{horizon}_T{n_diffusion_steps}_d{discount}'
current_dir=os.getcwd()
#ponerle nombre al experimento
exp_name=f"H{horizon}_T{n_diffusion_steps}_d{discount}_s{args["scale"]}_b{args["batch_size"]}"

diffusion_loadpath=os.path.join(current_dir,args["logbase"], args["dataset_name"],diffusion_loadpath)

value_loadpath=os.path.join(current_dir,args["logbase"], args["dataset_name"],value_loadpath)

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

value_experiment = utils.load_diffusion(
    value_loadpath,
    epoch="latest", seed=args["seed"],
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
dataset.inference_mode()
#print(dataset.normalizer.params_dict)

value_dataset=value_experiment.dataset
value_dataset.inference_mode()

## initialize value guide
#value_function = value_experiment.ema

#rew_model=Walker2d_reward(dataset)
guide_config = utils.Config(RawValueGuide, model=rew_model, verbose=False) # value function
guide = guide_config()



logger_config = utils.Config(
    utils.Logger,
    logpath=savepath
)

logger = logger_config()

## policies are wrappers around an unconditional diffusion model and a value guide

policy_config = utils.Config(
    GuidedPolicy,
    guide=guide,
    scale=args["scale"],  # scale sampling kwargs
    diffusion_model=diffusion,
    dataset=dataset,
    preprocess_fns=args["preprocess_fns"],
    ## sampling kwargs
    sample_fn=utils.Config(args["sample_fn"]),
    n_guide_steps=args["n_guide_steps"],
    t_stopgrad=args["t_stopgrad"],
    scale_grad_by_std=args["scale_grad_by_std"],
    verbose=args["verbose"],
)

policy = policy_config()
#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#


# si hacemos que dataset sea una instancia de value dataset, podemos obtener el score

env = dataset.minari_dataset.recover_environment(render_mode="rgb_array_list") # probar con rgb array list
#env = gym.make("HalfCheetah-v5",render_mode="rgb_array_list")

seed = int(datetime.now().timestamp())
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
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args["batch_size"], verbose=args["verbose"])
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

#score = value_dataset.normalize_value(total_reward)  # ojo con esto, dataset tiene que ser una instancia de value dataset y estar normed, probable recurrir a infos y tilizar datos externos...

## write results to json file at `args.savepath`
logger.finish(t, total_reward, terminated, diffusion_experiment, value_experiment,seed,args["scale"],args["batch_size"])
