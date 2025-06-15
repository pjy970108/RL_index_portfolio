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
from enviroment import Stock_Env
from ptflops import get_model_complexity_info  # 상단에 추가

import wandb 
# 논문 설정 기반 초기 sweep 구성
# --- Sweep 설정 ---
# --- Sweep 설정 ---
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'mean_reward', 'goal': 'maximize'},
    'parameters': {
        # 'lr_actor': {'values': [1e-6, 5e-6, 1e-5]},
        'lr_actor': {'values': [1e-5]},
        # 'clip_grad': {'values': [0.5, 0.1]},
        'epsilon': {'values': [0.1]},
        'beta': {'values': [0.0]},
        'mu': {'values': [50]},
        # 'num_group': {'values': [2, 4]},
        'num_steps': {'values': [10]},
        'reward_cond': {'values': ["sharpe", "combined_reward"]},
        "num_trajectories": {'values': [128]}
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
    with open("./portfolio_price_discrete_v4_small_filter/config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("./portfolio_price_discrete_v4_small_filter/config/test_config.yaml", "r") as f:
        test_config = yaml.safe_load(f)

    # 2. 디바이스 처리
    device = torch.device(config.get("device", "cuda:0"))
    config["device"] = device

    # 3. Train 데이터 로드
    index_before_df = pd.read_csv(config["data_paths"]["index_before_csv"], index_col=0)
    index_train_df = pd.read_csv(config["data_paths"]["index_train_csv"], index_col=0)
    index_train_df = pd.concat([index_before_df, index_train_df], axis=0)
    index_train_df.reset_index(drop=True, inplace=True)
    index_train_df.set_index("date", inplace=True)

    future_before_df = pd.read_csv(config["data_paths"]["future_before_csv"], index_col=0)
    future_train_df = pd.read_csv(config["data_paths"]["future_train_csv"], index_col=0)
    future_train_df = pd.concat([future_before_df, future_train_df], axis=0)
    future_train_df.reset_index(drop=True, inplace=True)
    future_train_df.set_index("date", inplace=True)



    #4. Test 데이터 로드
    index_train_for_valid = pd.read_csv(test_config["data_paths"]["index_train_csv"], index_col=0)
    index_test_df = pd.read_csv(test_config["data_paths"]["index_test_csv"], index_col=0)
    
    index_test_all_df = pd.concat([index_train_for_valid, index_test_df], axis=0)
    index_test_all_df.reset_index(drop=True, inplace=True)
    index_test_all_df.set_index("date", inplace=True)
    
    future_train_for_valid = pd.read_csv(test_config["data_paths"]["future_train_csv"], index_col=0)
    future_test_df = pd.read_csv(test_config["data_paths"]["future_test_csv"], index_col=0)
    future_test_all_df = pd.concat([future_train_for_valid, future_test_df], axis=0)
    future_test_all_df.reset_index(drop=True, inplace=True)
    future_test_all_df.set_index("date", inplace=True)

    test_traj_len ="1m"
    test_update_interval = test_config["update_interval_map"].get(test_traj_len, 60)
    test_config["update_interval"] = test_update_interval
    test_config["batch_samples"] = test_config["batch_samples"].get(test_traj_len, 5)

    # select_features = ['ret_norm',
    #    'close_norm', 'return_1m_norm', 'return_3m_norm', 'return_6m_norm',
    #    'return_12m_norm', 'return_avg_norm', 'mom_score_norm', 'vol_20_norm',
    #    'sharpe_252_norm', 'vol_252_norm', 'sortino_252_norm',
    #    'calmar_252_norm']
    # imp = pd.read_csv("data/portfolio_price/vaild_portfolio_price_v2.csv", index_col=0)
    # imp = imp[select_features]
    train_tensor = torch.load(config["data_paths"]["train_pt"], map_location=device)
    valid_tensor = torch.load(config["data_paths"]["valid_pt"], map_location=device)
    
    valid_env = Stock_Env(config=test_config, states=valid_tensor, index_df=index_test_all_df, future_df = future_test_all_df, eval=True)

    # for date in list(train_tensor.keys()):
    #     arr = train_tensor[date]
    #     train_tensor[date] = torch.tensor(arr, dtype=torch.float32).to(device)
    #     del arr
    
    # with open(config["data_paths"]["train_pkl"], "rb") as f:
    #     train_scaled = pickle.load(f)

    feature_dim = train_tensor.shape[-1]

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

    
    # traj_len = config.get("trajectory_length", "6m")
    traj_len ="5d"

    update_interval = config["update_interval_map"].get(traj_len, 60)
    config["update_interval"] = update_interval
    config["batch_samples"] = config["batch_samples"].get(traj_len, 5)
        
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
    model_path = f'./portfolio_price_discrete_v4_small_filter/model/{traj_len}/'
    os.makedirs(model_path, exist_ok=True)

    save_filename = (
        f'{traj_len}_actor_lr{config["lr_actor"]}'
        f'_clip_grad{config["clip_grad"]}_epslion_{config["epslion"]}'
        f'batch_sample_{config["batch_samples"]}_mu{config["mu"]}_num_group{config["num_trajectories"]}'
        f'_num_steps{config["num_steps"]}_reward_cond_{config["reward_cond"]}_grpo_model.pth'
    )
        
    config["save_path"] = os.path.join(model_path, save_filename)

    env = Stock_Env(config=config, states=train_tensor, index_df=index_train_df, future_df = future_train_df)
    
    # env.num_iterations = 2       # 전체 반복 횟수
    # env.num_steps = 1            # 한 iteration 내 GRPO rollout step 수
    # env.batch_samples = 2            # rollout group 수 (state 다양성 ↓)
    # # env.num_trajectories = 2     # 한 그룹 내 rollout 수
    # env.update_interval = 5      # 에피소드 길이 줄이기
    # env.mu = 3                   # update 반복 수
    # env.patience = 2             # early stopping 빠르게 확인
    # env.num_trajectories = 5
    
    
    
    agent = PPO(env)
    wrapped_model = ForwardableActorCritic(agent.policy)

    # grpo_model_stats(wrapped_model, input_shape=(1, env.feature_dim*env.asset_dim))  # ← eval 호출

    agent = optimize_model_memory(agent)
    agent.policy.to("cpu")
    env.states = env.states.to("cpu")

    trained_agent = train_with_grpo(agent=agent, env=env, valid_env = valid_env)
    final_path = os.path.join(model_path, "final_" + save_filename)

    torch.save(trained_agent.policy.state_dict(), final_path)

    wandb.finish()

if __name__ == "__main__":
    set_seed(42)
    sweep_id = wandb.sweep(sweep_config, project="grpo_portfolio_5d_discrete_v4_small_filter")
    wandb.agent(sweep_id, function=main_wandb)

    
