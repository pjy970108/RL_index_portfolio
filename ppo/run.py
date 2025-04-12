import torch
import copy
import pandas as pd
from TARN_PPO_daily.enviroment import EnvConfig, Stock_Env
import numpy as np
from TARN_PPO_daily.agent import PPO, EarlyStopping
import copy
from tqdm import tqdm
import wandb
import time


def train_daily_env(train_dataset, config):
    """
    step별로 학습하는 코드입니다.
    """
    env =Stock_Env(config)
    agent = PPO(env)
    early_stopping_reward = EarlyStopping(patience=config.patience, min_delta=0.0001)
    print("학습 환경 및 에이전트 생성 완료")
    print("학습 시작")
    for epoch in tqdm(range(config.epochs + 1)):
        _ = env.reset()

    



# def train_main_sub_env(train_tensor, train_dataset, train_data_episode, hyperparameters, config):
def train_main_sub_env(train_dataset, config):
    """
    메인 에피소드 : 전체 학습과정
    서브 에피소드 : 전체 학습과정 중 일부 학습과정(window size 1로하여 rolling하면서 학습되는 과정)
    """
    print("학습 환경 및 에이전트 생성")
    # config = EnvConfig(train_tensor, train_dataset)
    # config = EnvConfig(train_tensor, train_dataset, train_data_episode, hyperparameters)
    main_episodes = len(train_dataset.keys())

    env =Stock_Env(config)
    agent = PPO(env)
    # Early stopping 초기화
    early_stopping_reward = EarlyStopping(patience=config.patience, min_delta=0.0001)

    print("학습 환경 및 에이전트 생성 완료")
    print("학습 시작")
    for epoch in tqdm(range(config.epochs + 1)):
        # 돌때 마다 state 초기화
        _ = env.reset()
        # state = torch.tensor(state, dtype=torch.float32).to(config.device)
        main_episode_rewards = []
        global_time_step = 0  # 전체 타임스텝 카운트

        for main_episode in range(main_episodes):
            print(f"메인 에피소드 {main_episode} 시작")
            env.current_idx = main_episode  # Main Episode 시작 인덱스
            # env.current_idx = 7  # Main Episode 시작 인덱스

            state = env.window_slice() # 스테이트 짤림, day짤림, max_step 조정됨.
            state = state.permute(1, 2, 0)
            state = torch.tensor(state, dtype=torch.float32).to(config.device)
            total_sub_reward = []
            # time_step = 0
            # sub_epoch_reward = 0
            
            time_step = 0
       
            start_time = time.time()

            for _ in range(env.max_step):
                action, att = agent.select_action(state)
                next_state, reward, done, invest_money, reward_dict = env.step(action)
                reward = torch.tensor(reward, dtype=torch.float32).to(config.device)
                # 에이전트 버퍼에 데이터 저장
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
                next_state = next_state.permute(1, 2, 0)

                state = torch.tensor(next_state, dtype=torch.float32).to(config.device)
                # sub_epoch_reward += reward.item()
                total_sub_reward.append(reward.item()) # # step별 reward 저장
                if reward > 20:
                    print(f"reward: {reward}, invest_money: {invest_money}, reward_dict: {reward_dict}")
                wandb.log({"sub_episode_reward": reward,  "time_step": global_time_step})
                time_step += 1
                global_time_step += 1  # 전체 타임스텝 증가

                # 정책 업데이트 주기에 도달하면 업데이트 실행
                if time_step % env.update_interval == 0:
                    torch.set_grad_enabled(True)
                    agent.update()
                    torch.set_grad_enabled(False)

                    time_step = 0
                    current_loss  = agent.get_last_metrics()  # PPO에서 구현 필요
                    # wandb.log({"agent_loss_per_sub_episode": current_loss})
                

                
                if done: # 서브에피소드 종료
                    end_time = time.time()
                    # print(f"메인 에피소드 {main_episode} 종료, 소요시간: {end_time - start_time:.2f}초")
                    break

            # 메인 에피소드의 보상 기록 [[sub_reward1], [sub_reward2], ...]
            main_episode_rewards.append(np.mean(total_sub_reward))
            # 정책 업데이트 주기에 도달하면 업데이트 실행
            # if time_step % config.update_interval == 0:
            # 메인에피소드가 끝날때마다 업데이트
            # torch.set_grad_enabled(True)
            # agent.update()
            # torch.set_grad_enabled(False)
            # time_step = 0
            # current_loss  = agent.get_last_metrics()  # PPO에서 구현 필요
            # wandb.log({"agent_loss_per_sub_episode": current_loss})

                    # early_stopping(avg_main_episode_reward)
        # if early_stopping.early_stop:
        #     print(f"Early stopping triggered at epoch {epoch}.")
        #     break
        agent.decay_action_std(env.action_std_decay_rate, env.min_action_std)
        
        avg_main_episode_reward = np.mean(main_episode_rewards)

        wandb.log({"main_episode_reward": avg_main_episode_reward, "epoch": epoch})
        # Early stopping 체크
        early_stopping_reward(avg_main_episode_reward)
        if early_stopping_reward.early_stop:
            print(f"Early stopping triggered by reward at epoch {epoch}. Avg Reward: {avg_main_episode_reward:.2f}")
            break

        # 주기적으로 로그 출력
        # # if epoch % config.log_interval == 0:
        # if epoch % 1 == 0:

        #     # 리스트 안에 리스트 각각 평균내어 메인 에피소드의 평균 보상 계산
        #     avg_reward = np.mean([np.mean(main_episode) for main_episode in main_episode_rewards])
        #     print(f"Epoch {epoch}/{config.epochs} \t Average Reward: {avg_reward:.2f}")
        
            # 주기적으로 모델 저장
        if epoch % 10 == 0:
            agent.save()
            print(f"Model saved at epoch {epoch}")
    agent.save()
    print("학습 종료")
    
    return agent
    

def eval_env(env, agent):
    
    """
    테스트 기간동안 평가하기 위한 코드입니다.
    """
    # print("평가 환경 생성")
    # config = EnvConfig(test_tensor, test_dataset)
    # env = copy.deepcopy(Stock_Env(test_tensor, test_dataset, config))
    # print("평가 시작")
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(env.device)
    rewards = []
    asset_weight_li = []
    top3_action_li = []
    agent.policy.eval()         # 평가 모드 전환
    agent.policy_old.eval()     # (필요하다면) old policy도 평가 모드로
    
    for _ in range(env.max_step):
        state = state.permute(1, 2, 0)
        state = torch.tensor(state, dtype=torch.float32).to(env.device)
        with torch.no_grad():
            action, attn_score = agent.select_action(state,  deterministic=True)
            print(attn_score.shape)
        next_state, rewards, done, n_rewards,  all_weight_invest_rewards, asset_weight_dict, top3_action = env.eval_step(action)
        asset_weight_li.append(asset_weight_dict)
        top3_action_li.append(top3_action) 
        state = torch.tensor(next_state, dtype=torch.float32).to(env.device)
        # rewards.append(reward_dict["return"])
        if done:
            break
    return rewards, n_rewards, all_weight_invest_rewards, asset_weight_li, top3_action_li


if __name__ == "__main__":
    import torch
    import copy
    import pandas as pd
    from enviroment import EnvConfig, Stock_Env
    import numpy as np
    from agent import PPO
    import pickle
    # pickle 파일을 불러오는 방법
    with open("./data/train_real_data.pkl", "rb") as f:
        train_dataset_episode = pickle.load(f)
    
    
    train_dataset = pd.read_csv("./data/train_data.csv", index_col=0)

    
    feature_dim = 35


    train_scaled_tensor = torch.load("./data/all_episodes.pt")
    
    # --- Hyperparameters ---
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
            "final_conv_config": {"in_channels" : 16, "out_channels": 16, "kernel_size": 31, "stride": 1, "padding": 0}
        }

    hyperparameters = get_hyperparameters(feature_dim)

    

    train_agent = train_main_sub_env(train_scaled_tensor, train_dataset, train_dataset_episode, hyperparameters)
    
    # rewards = eval_env(test_tensor, test_dataset, train_agent)
