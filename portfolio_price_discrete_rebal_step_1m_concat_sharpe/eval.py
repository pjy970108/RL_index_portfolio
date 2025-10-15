import sys
import os
import pandas as pd
import numpy as np
import torch
import copy
import pandas as pd
import torch
import yaml
from enviroment import Stock_Env
from agent import PPO
from grpo import test_grpo



if __name__ == "__main__":
    def set_seed(seed):
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경 고려
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    SEED = 42
    set_seed(SEED)
    
    with open("./portfolio_price_discrete_rebal_step_1m_concat_asset/config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    train_data_path = "data/train_v3.csv"
    test_data_path = "data/test_v3.csv"
    test_pt = "data/portfolio_price/concat_portfolio_test_monthly_v1.pt"
    
    device = torch.device(config.get("device", "cuda:0"))
    config["device"] = device
    
    index_train_for_valid = pd.read_csv(config["data_paths"]["index_train_csv"], index_col=0)
    index_test_df = pd.read_csv(config["data_paths"]["index_test_csv"], index_col=0)
    
    index_test_all_df = pd.concat([index_train_for_valid, index_test_df], axis=0)
    index_test_all_df.reset_index(drop=True, inplace=True)
    index_test_all_df.set_index("date", inplace=True)
    
    future_train_for_valid = pd.read_csv(config["data_paths"]["future_train_csv"], index_col=0)
    future_test_df = pd.read_csv(config["data_paths"]["future_test_csv"], index_col=0)
    future_test_all_df = pd.concat([future_train_for_valid, future_test_df], axis=0)
    future_test_all_df.reset_index(drop=True, inplace=True)
    future_test_all_df.set_index("date", inplace=True)
    
    
    test_tensor = torch.load(config["data_paths"]["test_pt"], map_location=device)
    traj_len ="1m"
    update_interval = config["update_interval_map"].get(traj_len, 60)
    config["update_interval"] = update_interval
    config["batch_samples"] = config["batch_samples"].get(traj_len, 5)
    config["TRADE_START_DATE"] = "2019-01-01"
    config["TRADE_END_DATE"] = "2024-12-31"
    env = Stock_Env(config=config, states=test_tensor, index_df=index_test_all_df, future_df = future_test_all_df, eval=True)
    
    agent = PPO(env, eval=True)
    agent.load(config["data_paths"]["model_path"])
    model_return, strategy_returns, weight_rebal, strategy_weight = test_grpo(env, agent)
    
