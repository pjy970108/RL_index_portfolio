import torch 
import os
import sys

import numpy as np
import random 
import yaml
import pandas as pd
import pickle

from grpo import train_with_grpo, optimize_model_memory
from agent import PPO, ForwardableActorCritic
import copy
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "base_line_model/Task_1_FinRL_DeepSeek_Stock")))
# print("Current working directory:", os.getcwd())
# print(sys.path)
from grpo.direct.enviroment import Stock_Env
from ptflops import get_model_complexity_info  # 상단에 추가

import wandb 
# 논문 설정 기반 초기 sweep 구성
# --- Sweep 설정 ---
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'mean_reward', 'goal': 'maximize'},
    'parameters': {
        # 'lr_actor': {'values': [1e-6, 5e-6, 1e-5]},
        'lr_actor': {'values': [1e-6]},
        # 'clip_grad': {'values': [0.5, 0.1]},
        'epsilon': {'values': [0.1]},
        'beta': {'values': [0.04]},
        'mu': {'values': [30]},
        # 'num_group': {'values': [2, 4]},
        'num_steps': {'values': [5, 10, 15]},
        'reward_cond': {'values': ["sharpe", "combined_reward"]},
        "num_trajectories": {'values': [32, 64]}
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def get_hyperparameters(feature_dim):
    return {
        "dcc_dropout": 0.2,
        "sac_dropout": 0.01,
        "sac_heads": 2,
        "ddc_configs": [
            {"in_channels": feature_dim, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "sac_scale": 4**0.5, "residual_out_channels": 8, "residual_kernal": 1},
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2, "sac_scale": 8**0.5, "residual_out_channels": 16, "residual_kernal": 1},
            {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 4, "dilation": 4, "sac_scale": 8**0.5}
        ],
        "final_conv_config": {"in_channels": 16, "out_channels": 8, "kernel_size": 20, "stride": 1, "padding": 0}
    }


def grpo_model_stats(model, input_shape):
    model.eval()
    with torch.cuda.device(0 if torch.cuda.is_available() else -1):
        flops, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=False
        )
        print(f"\n🔍 [GRPO POLICY MODEL STATS]")
        print(f"FLOPs : {flops}")
        print(f"Params: {params}")



def main_wandb():
    wandb.init(project="GRPO-sweep", config=sweep_config['parameters'])
    config_wandb = wandb.config
    
    # 1. config 파일 로드
    with open("./grpo/config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. 디바이스 처리
    device = torch.device(config.get("device", "cuda:0"))
    config["device"] = device

    # 3. 데이터 로드
    before_df = pd.read_csv(config["data_paths"]["before_csv"], index_col=0)
    train_df = pd.read_csv(config["data_paths"]["train_csv"], index_col=0)
    train_df = pd.concat([before_df, train_df], axis=0)
    train_df.reset_index(drop=True, inplace=True)
    train_df.set_index("date", inplace=True)

    train_tensor = torch.load(config["data_paths"]["train_pt"], map_location=device)
    
    # for date in list(train_tensor.keys()):
    #     arr = train_tensor[date]
    #     train_tensor[date] = torch.tensor(arr, dtype=torch.float32).to(device)
    #     del arr
    
    # with open(config["data_paths"]["train_pkl"], "rb") as f:
    #     train_scaled = pickle.load(f)

    feature_dim = train_tensor.shape[-1]
    hyperparameters = get_hyperparameters(feature_dim)

    # 4. wandb sweep 결과를 config에 반영
    config.update({
        "lr_actor": config_wandb.lr_actor,
        # "clip_grad": config_wandb.clip_grad,
        "epslion": config_wandb.epsilon,
        "beta": config_wandb.beta,
        "mu": config_wandb.mu,
        # "num_group": config_wandb.num_group,
        "num_steps": config_wandb.num_steps,
        "reward_cond": config_wandb.reward_cond,
        "num_trajectories" : config_wandb.num_trajectories
    })

    traj_len ="6m"

    traj_len = config.get("trajectory_length", "3m")
    update_interval = config["update_interval_map"].get(traj_len, 60)
    config["update_interval"] = update_interval
    config["batch_samples"] = config["batch_samples"].get(traj_len, 5)
    
    # config.num_iterations = 2

    
    # config["num_trajectories"] = 5
    # config["batch_samples"]  = 2
    # config.num_iterations = 1
    # config.num_steps = 1  # 튜닝 가능하게 변경
    # config.mu = 1
    # config.beta = config_wandb.beta
    # config.epsilon = config_wandb.epsilon
    # config.clip_grad = config_wandb.clip_grad
    # config.lr_actor = config_wandb.lr_actor
    # config.num_group = 1
    
    # GRPO 학습 하이퍼파라미터 정의
    # num_iterations = 
    # num_prompts = 16
    # num_generations = 8
    # mu = 3
    # beta = 0.01
    # epsilon = 0.2
    # lr = 5e-5
    # 경로 없으면 경로 생성하는 코드
    # 6. 모델 저장 경로 설정
    model_path = f'./grpo/model/{traj_len}/'
    os.makedirs(model_path, exist_ok=True)

    save_filename = (
        f'{traj_len}_actor_lr{config["lr_actor"]}'
        f'_clip_grad{config["clip_grad"]}_epslion_{config["epslion"]}'
        f'beta{config["beta"]}_mu{config["mu"]}_num_group{config["num_group"]}'
        f'_num_steps{config["num_steps"]}_reward_cond_{config["reward_cond"]}_batch_samples_{config["batch_samples"]}_TARN_grpo_model.pth'
    )
        
    config["save_path"] = os.path.join(model_path, save_filename)

    env = Stock_Env(config=config, states=train_tensor, real_df=train_df, hyperparameters=hyperparameters)
    
    
    # env.num_iterations = 2       # 전체 반복 횟수
    # env.num_steps = 1            # 한 iteration 내 GRPO rollout step 수
    # env.batch_samples = 20            # rollout group 수 (state 다양성 ↓)
    # env.num_trajectories = 2     # 한 그룹 내 rollout 수
    # env.update_interval = 5      # 에피소드 길이 줄이기
    # env.mu = 1                   # update 반복 수
    # env.patience = 2             # early stopping 빠르게 확인
    # env.num_trajectories_map = 5
    
    
    
    agent = PPO(env)
    wrapped_model = ForwardableActorCritic(agent.policy)

    grpo_model_stats(wrapped_model, input_shape=(22, 13, 20))  # ← eval 호출

    agent = optimize_model_memory(agent)

    trained_agent = train_with_grpo(agent=agent, env=env)
    final_path = os.path.join(model_path, "final_" + save_filename)

    torch.save(trained_agent.policy.state_dict(), final_path)

    wandb.finish()

if __name__ == "__main__":
    set_seed(42)
    sweep_id = wandb.sweep(sweep_config, project="grppo_index_3m_v1")
    wandb.agent(sweep_id, function=main_wandb)

    
