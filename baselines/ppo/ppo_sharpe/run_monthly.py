import torch
import copy
import pandas as pd
import numpy as np
try:
    from .enviroment import Stock_Env
    from .agent import PPO, EarlyStopping
except ImportError:
    from enviroment import Stock_Env
    from agent import PPO, EarlyStopping
import copy
from tqdm import tqdm
import wandb
import time 
from backtesting_all_asset_monthly import eval_partial_backtests_v2  # 함수만 명시적으로 import
import eval_metric


# def train_main_sub_env(train_tensor, train_dataset, train_data_episode, hyperparameters, config):
def train_main_sub_env(agent,  env, valid_env):
    """
    메인 에피소드 : 전체 학습과정
    서브 에피소드 : 전체 학습과정 중 일부 학습과정(window size 1로하여 rolling하면서 학습되는 과정)
    """
    print("학습 환경 및 에이전트 생성")
    # config = EnvConfig(train_tensor, train_dataset)
    # config = EnvConfig(train_tensor, train_dataset, train_data_episode, hyperparameters)

    # Early stopping 초기화
    best_avg_reward = -float("inf")
    patience_counter = 0
    print("학습 환경 및 에이전트 생성 완료")
    print("학습 시작")
    for epoch in tqdm(range(env.epochs + 1)):
        # 돌때 마다 state 초기화
        # state = torch.tensor(state, dtype=torch.float32).to(config.device)
        main_episode_rewards = []
        global_time_step = 0  # 전체 타임스텝 카운트

        for episode in range(env.batch_samples):
            state = env.reset()
            # env.current_idx = 7  # Main Episode 시작 인덱스
            state = state.reshape(1, -1)
            state = torch.tensor(state, dtype=torch.float32).to(env.device)
            total_sub_reward = []
            # time_step = 0
            # sub_epoch_reward = 0
            
            time_step = 0
       
            start_time = time.time()
            with torch.no_grad():
                for _ in range(env.update_interval//env.window_size):
                    actions = agent.select_action(state)
                    # actions = actions.reshape(1, -1)
                    next_state, reward, done, reward_dict = env.step(actions)
                    reward = torch.tensor(reward, dtype=torch.float32).to(env.device)
                # 에이전트 버퍼에 데이터 저장
                    agent.buffer.rewards.append(reward)
                    agent.buffer.is_terminals.append(done)
                    next_state = next_state.reshape(1, -1)
                    state = next_state.clone()

                # sub_epoch_reward += reward.item()
                    total_sub_reward.append(reward.item()) # # step별 reward 저장


            # 정책 업데이트 주기에 도달하면 업데이트 실행
            if episode == (env.batch_samples-1):
                torch.set_grad_enabled(True)
                avg_policy_loss, avg_value_loss, avg_entropy_bonus, avg_total_loss= agent.update(env.mini_batch_size)
                torch.set_grad_enabled(False)

                time_step = 0
                current_loss  = agent.get_last_metrics()  # PPO에서 구현 필요
                # wandb.log({"agent_loss_per_sub_episode": current_loss})
            
                agent.policy.eval()
                with torch.no_grad():
                    state = valid_env.reset()
                    state = state.reshape(1, -1)
                    state = torch.tensor(state, dtype=torch.float32)
                    strategy_returns, strategy_weights = eval_partial_backtests_v2(index_long=valid_env.index_df,
                                                                                future_long = valid_env.future_df,
                                                                                    equal_long =  valid_env.all_df,

                            trading_data=(valid_env.days[0], valid_env.days[-1]),
                            top_n=valid_env.top_n,
                            look_backs=valid_env.look_back,
                            rebalance_every=valid_env.window_size,
                            cost=valid_env.cost,
                            top_pct = valid_env.top_pct,
                            risk_coefficient=valid_env.risk_coefficient)
                    
                    model_return = pd.DataFrame()
                    for _ in range(valid_env.max_step):
                        action, _, _ = agent.policy.act(state, deterministic=True)
                        
                        next_state, combined_returns, done, _, action_dict = valid_env.eval_step(action.cpu().numpy(), strategy_returns)
                        next_state = next_state.reshape(1, -1)
                        state = next_state.clone()
                        model_return = pd.concat([model_return, combined_returns], axis=0)

                        if done:
                            break
                    
                    valid_sharpe = eval_metric.calculate_sharpe_ratio(model_return, risk_free_rate=valid_env.risk_free_rate, annual_factor=valid_env.annual_factor)
                wandb.log({
                    "valid_reward": valid_sharpe,
                    "train_policy_loss": float(avg_policy_loss),
                    "train_mean_reward": float(avg_value_loss),
                    "train_avg_entropy_bonus": avg_entropy_bonus,
                    "train_avg_total_loss": avg_total_loss})
                agent.policy.train()            
 

            # 각 기간의 평균 보상
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
        
        avg_main_episode_reward = np.mean(main_episode_rewards)

        wandb.log({"train_episode_reward": avg_main_episode_reward})
        # Early stopping 체크
        if (epoch + 1) % 2 == 0:
            torch.save(agent.policy.state_dict(), env.save_path)
            wandb.log({"model_saved_at": env.save_path})
            print(f"💾 Saved model at: {env.save_path}")

                
        # Early stopping 체크
        # avg_valid_reward =valid_sharpe  # ✅ valid reward
        if valid_sharpe > best_avg_reward:
            best_avg_reward = valid_sharpe
            patience_counter = 0
            print(f"  ✅ New best reward: {best_avg_reward:.4f}")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement. Patience: {patience_counter}/{env.patience}")
            if patience_counter >= env.patience:
                print("\n🛑 Early stopping triggered.")
                torch.save(agent.policy.state_dict(), env.save_path)

                break

        # 주기적으로 로그 출력
        # # if epoch % config.log_interval == 0:
        # if epoch % 1 == 0:

        #     # 리스트 안에 리스트 각각 평균내어 메인 에피소드의 평균 보상 계산
        #     avg_reward = np.mean([np.mean(main_episode) for main_episode in main_episode_rewards])
        #     print(f"Epoch {epoch}/{config.epochs} \t Average Reward: {avg_reward:.2f}")
        
            # 주기적으로 모델 저장
    print("\n[GRPO Training Finished]")
    torch.save(agent.policy.state_dict(), env.save_path)
    
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
    state = state.reshape(1, -1)
    state = torch.tensor(state, dtype=torch.float32)
    strategy_returns, strategy_weights = eval_partial_backtests_v2(index_long=env.index_df,
                                                                    future_long = env.future_df,
                                                                    equal_long =  env.all_df,

            trading_data=(env.days[0], env.days[-1]),
            top_n=env.top_n,
            look_backs=env.look_back,
            rebalance_every=env.window_size,
            cost=env.cost,
            top_pct = env.top_pct,
            risk_coefficient=env.risk_coefficient)
    model_return = pd.DataFrame()
    weight_model = pd.DataFrame()

    
    for _ in range(env.max_step):
        action, _, _ = agent.policy.act(state, deterministic=True)
        next_state, combined_returns, done, _, action_dict = env.eval_step(action.cpu().numpy(), strategy_returns)
        weight_model = pd.concat([weight_model, pd.DataFrame(action_dict).T], axis=0)
        next_state = next_state.reshape(1, -1)
        state = next_state.clone()
        model_return = pd.concat([model_return, combined_returns], axis=0)

        # rewards.append(reward_dict["return"])
        if done:
            break
    return model_return, strategy_returns, weight_model, strategy_weights


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
