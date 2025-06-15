import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

root_list = ["/data_preprocessing/", "/TARN_MAPPO"]
for root in root_list:
    sys.path.append(os.getcwd() + root)
import pandas as pd
import numpy as np
from data_preprocessing.preprocessing_config import *
from utils import *
import dynamic_portfolio as dp
import data_preprocessing.technical_indicators as ti
import data_preprocessing.main_preprocessing as preprop
from TARN_MAPPO.run_rolling import *
import copy
from TARN_MAPPO.enviroment_rolling import EnvConfig, Stock_Env
import wandb
import pickle
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경 고려
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")



sweep_config = {
    "method": "grid",  # ✅ Bayesian Optimization 사용
    "metric": {"name": "average_reward", "goal": "maximize"},  # 평가 기준 설정
    "parameters": { 
        "seed": {"values": [42]},
        "reward_cond" : {"values": ["combined"]},
        # "lr_actor": {"values": [1e-3, 0.01, 0.1]},
        "lr_actor": {"values": [1e-4, 1e-3]},

        "lr_critic": {"values": [1e-5, 1e-4]},
        "gamma": {"values": [0.99]},
        # "gamma": {"values": [0.99]},

        "eps_clip": {"values": [0.1, 0.2, 0.3]},
        # "entropy_weight" : {"values": [0.01, 0.1, 0.5]},
        "entropy_weight" : {"values": [0.1, 0.01]},
        "num_agents": {"values": [4, 3, 2]},
        "K_epochs": {"values": [80]} # Deep Reinforcement Learning Robots for Algorithmic Trading: Considering Stock Market Conditions and U.S. Interest Rates

    },
}

def main():
    
    wandb.init(project="rl-project")
    # wandb.init(mode="online", settings=wandb.Settings(start_method="thread"))
    wandb.define_metric("epoch")
    wandb.define_metric("main_episode_reward", step_metric="epoch")
    wandb.define_metric("sub_episode_reward", step_metric="time_step")
    
    
    with open("./data/train_daily_v4.pkl", "rb") as f:
        train_scaled = pickle.load(f)
    print("현재 작업 디렉토리:", os.getcwd())

    train_df = pd.read_csv("data/train_v4.csv", index_col=0)
    train_df.set_index("date", inplace=True)
    feature_dim = 15
    gpu_name = 1  # 0번 GPU를 의미

    if gpu_name is not None:
        torch.cuda.set_device(gpu_name)
        device = torch.device(f"cuda:{gpu_name}")
    else:
        device = torch.device("cpu")
    

        
    # train_scaled_tensor = torch.load("data/daily_data_v3.pt")
    train_scaled_tensor = torch.load("data/daily_data_v4.pt", map_location=device)
    
    
    def get_hyperparameters(feature_dim):
        return {
            "dcc_dropout": 0.2,
            "sac_dropout": 0.01,
            "sac_heads": 2,
            "ddc_configs": [
                {"in_channels": feature_dim, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": (3-1)//2, "dilation": 1, "sac_scale": 4**0.5, "residual_out_channels" : 8, "residual_kernal": 1},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2, "sac_scale": 8**0.5, "residual_out_channels" : 16, "residual_kernal": 1 },
                {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 4, "dilation": 4, "sac_scale": 8**0.5}
            ],
            "final_conv_config": {"in_channels" : 16, "out_channels": 8, "kernel_size": 20, "stride": 1, "padding": 0}
        }
    hyperparameters = get_hyperparameters(feature_dim)
    trajectory_lenght =  "3m" #"1y", "6m"
    config = EnvConfig(train_scaled_tensor, train_df, train_scaled, hyperparameters, trajectory_lenght)
    config.device = device
    config.lr_actor = wandb.config.lr_actor
    config.lr_critic = wandb.config.lr_critic
    # config.action_std_init = wandb.config.action_std_init
    config.gamma = wandb.config.gamma
    config.eps_clip = wandb.config.eps_clip
    config.K_epochs = wandb.config.K_epochs
    config.reward_cond = wandb.config.reward_cond
    config.entropy_weight = wandb.config.entropy_weight

    # config.update_interval = wandb.config.update_interval
    # config.patience = wandb.config.patience
    config.num_agents = wandb.config.num_agents
    # config.epochs = wandb.config.epochs
    # 경로 없으면 경로 생성하는 코드
    model_path = f'./TARN_MAPPO/model/{trajectory_lenght}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    config.save_path = f'{model_path}/{trajectory_lenght}_risk_scaled_data_actor_num_agent_{config.num_agents}_lr{config.lr_actor}_critic_lr{config.lr_critic}_gamma{config.gamma}_eps_clip{config.eps_clip}_reward_cond_{config.reward_cond}_entropy_weight{config.entropy_weight}__k_epoch{config.K_epochs}_TARN_mappo_model.pth'
    train_agent = train_daily_env(train_scaled_tensor, config)



if __name__ == "__main__":
    
    SEED = 42
    set_seed(SEED)
    sweep_id = wandb.sweep(sweep_config, project="TARN-MAPPO-project_daily_3m_ippo_v1")
    print(f"Sweep ID: {sweep_id}")  # Sweep ID 출력
    wandb.agent(sweep_id, function=main)  # count는 반복 횟수
