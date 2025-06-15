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
    
    with open("./portfolio_price_softmax/config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    train_data_path = "data/train_v2.csv"
    test_data_path = "data/train_v2.csv"
    test_pt = "data/portfolio_price/portfolio_test.pt"
    
    device = torch.device(config.get("device", "cuda:0"))
    config["device"] = device
    
    train_df = pd.read_csv(config["data_paths"]["train_csv"], index_col=0)
    test_df = pd.read_csv(config["data_paths"]["test_csv"], index_col=0)
    test_all_df = pd.concat([train_df, test_df], axis=0)
    test_all_df.reset_index(drop=True, inplace=True)
    test_all_df.set_index("date", inplace=True)
    
    test_tensor = torch.load(config["data_paths"]["test_pt"], map_location=device)
    traj_len ="1m"
    update_interval = config["update_interval_map"].get(traj_len, 60)
    config["update_interval"] = update_interval
    config["batch_samples"] = config["batch_samples"].get(traj_len, 5)
    config["TRADE_START_DATE"] = "2017-01-03"
    config["TRADE_END_DATE"] = "2023-12-29"
    env = Stock_Env(config=config, states=test_tensor, real_df=test_all_df, eval=True)
    
    agent = PPO(env, eval=True)
    agent.load(config["data_paths"]["model_path"])
    model_return, strategy_returnsm, weight_rebal, strategy_weight = test_grpo(env, agent)
    
