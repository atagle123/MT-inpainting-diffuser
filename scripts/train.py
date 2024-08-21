import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffuser.utils.training import Trainer
from diffuser.utils.arrays import report_parameters,batchify
from diffuser.utils.config import Config
from diffuser.utils.setup import load_experiment_params,set_seed
import torch
import wandb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dataset="walker2d"

args=load_experiment_params(f"logs/configs/{dataset}/{dataset}-medium-expert-v2/configs_diffusion.txt")

set_seed(args["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {device}")

## serialization
current_dir=os.getcwd()

exp_name=f"H{args["horizon"]}_T{args["n_diffusion_steps"]}"
savepath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"diffusion", exp_name)
os.makedirs(savepath, exist_ok=True)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = Config(
    args["dataset"],
    savepath=(savepath, 'dataset_config.pkl'),
    env_name=args["dataset_name"],
    horizon=args["horizon"],
    normalizer=args["normalizer"],
    preprocess_fns=args["preprocess_fns"],
    max_path_length=args["max_path_length"]
)
dataset = dataset_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model_config = Config(
    args["model"],
    savepath=(savepath, 'model_config.pkl'),
    device=device,
    horizon=args["horizon"],
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args["dim_mults"],
    attention=args["attention"],
)

diffusion_config = Config(
    args["diffusion_model"],
    savepath=(savepath, 'diffusion_config.pkl'),
    horizon=args["horizon"],
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args["n_diffusion_steps"],
    loss_type=args["loss_type"],
    clip_denoised=args["clip_denoised"],
    predict_epsilon=args["predict_epsilon"],
    ## loss weighting
    action_weight=args["action_weight"],
    loss_weights=args["loss_weights"],
    loss_discount=args["loss_discount"],
    device=device,
)

model = model_config()
diffusion = diffusion_config(model)


trainer_config = Config(
    Trainer,
    savepath=(savepath, 'trainer_config.pkl'),
    train_batch_size=args["batch_size"], # batch size number
    train_lr=args["learning_rate"],
    gradient_accumulate_every=args["gradient_accumulate_every"],
    ema_decay=args["ema_decay"],
    sample_freq=args["sample_freq"],
    save_freq=args["save_freq"],
    label_freq=int(args["n_train_steps"] // args["n_saves"]),
    results_folder=savepath,
    wandb_log=args["wandb_log"]
)

trainer = trainer_config(diffusion, dataset)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args["n_train_steps"] // args["n_steps_per_epoch"])
wandb_name=f"{args["dataset_name"]}_diffusion_H{args["horizon"]}_T{args["n_diffusion_steps"]}"

if args["wandb_log"]:
    wandb.init(
        project='Diffusion_RL_thesis',
        name=wandb_name,
        monitor_gym=True,
        save_code=True)
    
    wandb.config=args


for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {savepath}')
    trainer.train(n_train_steps=args["n_steps_per_epoch"])

wandb.finish()