import copy 
import wandb
import torch 
import random 
import numpy as np 
import torch.nn as nn
from contextlib import nullcontext
from agent import PPO
from enviroment import Stock_Env
import wandb
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import sys
from backtesting_all_asset import eval_partial_backtests_v2  # 함수만 명시적으로 import
import pandas as pd
import eval_metric


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return {
            "states": traj["states"],
            "actions": traj["actions"],
            "log_probs": traj["log_probs"],
            # "ref_log_probs": traj["ref_log_probs"],
            "rewards": traj["rewards"],
            "advantage": traj["advantage"],
        }


def get_memory_usage():
    """Get current GPU memory usage in a human-readable format."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CUDA not available"

        
# grpo가 환경에서 실행한 하나의 Trajectory
# 그 후 몇개의 rollout 설정해

def generate_single_group(idx, model_state, env):
    group = []
    state = env.reset()
    state = state.reshape(1, -1)
    state = torch.tensor(state, dtype=torch.float32)
    model = PPO(env)
    model.policy_old.to("cpu")

    model.policy_old.load_state_dict(model_state)

    for _ in range(env.num_trajectories):
        trajectory = {"states": [], "actions": [], "log_probs": [], "rewards": []}

        with torch.no_grad():
            for _ in range(env.update_interval//env.window_size):
                actions, logprob = model.policy_old.act(state)

                actions_np = actions.detach().cpu().numpy()
                next_state, reward, done, _ = env.step(actions_np, group_reset=True)

                trajectory["states"].append(state.clone().cpu())
                actions_np = actions_np.reshape(-1)
                trajectory["actions"].append(torch.tensor(actions_np, dtype=torch.float32))
                trajectory["log_probs"].append(logprob.detach())
                trajectory["rewards"].append(torch.tensor(reward, dtype=torch.float32))

                next_state = next_state.reshape(1, -1)
                state = next_state.clone()
                if done:
                    print(env.idx)
                    state = env.states[env.idx]
                    state = state.reshape(1, -1)
                    state= state.clone()
                    pass

        for key in trajectory:
            trajectory[key] = torch.stack(trajectory[key])
        group.append(trajectory)
    return group


# --- PARALLEL ROLLOUT ENTRY ---
def generate_rollout_data(model, env):
    DEBUG_MODE = sys.gettrace() is not None  # VSCode 디버깅 중이면 True

    n_jobs = 1 if DEBUG_MODE else env.num_workers
    print(f"Using {n_jobs} parallel workers for rollout generation.")
    
    model_state = model.policy_old.state_dict()
    rollout_groups = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_group)(idx, model_state, env)
        for idx in range(env.batch_samples)
    )
    return rollout_groups


def grpo_update(agent, rollout_groups, env, optimizer):
    """
    전체 trajectory를 DataLoader로 미니배치 학습하고, 
    optimizer는 각 배치마다 step을 진행
    """
    all_trajectories = []
    norm_rewards_list = []

    # reward 정규화
    for group in rollout_groups:
        group_returns_list = []
        for traj in group:
            rewards = traj["rewards"]               # shape: [T]
            return_sum = rewards.sum()              # trajectory 전체 reward의 합
            group_returns_list.append(return_sum)   # 리스트에 추가
        
        group_returns = torch.stack(group_returns_list)
        # 평균과 표준편차 계산
        min_group = group_returns.min()
        max_group = group_returns.max()

        # trajectory별 advantage 계산
        for traj, return_sum in zip(group, group_returns_list):
            traj_advantage = (return_sum - min_group) / (max_group-min_group)
            t_len = traj["rewards"].shape[0]  # Trajectory 길이
            traj["advantage"] = traj_advantage.repeat(t_len)  # shape: [T]
            # all_trajectories.append(traj)  # ✅ trajectory 저장
        # # 전체 trajectory 리스트에 추가

        all_trajectories.extend(group)

    dataset = TrajectoryDataset(all_trajectories)
    loader = DataLoader(dataset, batch_size=env.mini_batch_size, shuffle=True)

    total_loss = 0
    all_rewards = []
    all_advantages = []

    for batch in loader:
        states = batch["states"].to(env.device, non_blocking=True)
        actions = batch["actions"].to(env.device, non_blocking=True)
        log_probs = batch["log_probs"].to(env.device, non_blocking=True)
        # ref_log_probs = batch["ref_log_probs"].to(env.device, non_blocking=True)
        # rewards = batch["rewards"].to(env.device, non_blocking=True)
        advantage = batch["advantage"].to(env.device, non_blocking=True)
        

        logprob_new, _ = agent.policy.evaluate(states, actions)
        logprob_old = log_probs.squeeze(-1)

        # logprob_ref = ref_log_probs.squeeze(-1)
        
        ratio = torch.exp(logprob_new - logprob_old)
        clipped_ratio = torch.clamp(ratio, 1 - env.epsilon, 1 + env.epsilon)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        # kl = torch.exp(logprob_ref - logprob_new) - (logprob_ref - logprob_new) - 1
        # loss_all = -surrogate + env.beta * kl
        loss_all = -surrogate

        loss = loss_all.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), env.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        all_rewards.append(batch["rewards"].mean(dim=1).mean())
        all_advantages.append(advantage.mean().item())
    avg_epoch_loss = total_loss / len(loader)

    return avg_epoch_loss, all_rewards, all_advantages



def train_with_grpo(agent, env, valid_env):
    
    best_avg_reward = -float("inf")
    patience_counter = 0
    total_steps = env.num_iterations * env.num_steps  # 전체 step 수=
    optimizer = torch.optim.Adam(agent.parameters(), lr=env.lr_actor)

    for global_step in range(total_steps):
        print(f"\n[Global Step {global_step+1}/{total_steps}]")
        
        with torch.no_grad():
            rollout_data = generate_rollout_data(
                agent,
                env
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # num_steps만큼 반복
        # batch samples 
        # sample g를 뽑음 
            
            # agent.policy_old.to(env.device)
            # ref_model.policy_old.to(env.device)            
        agent.policy.to(env.device)

        for grpo_iter in range(env.mu):
            print(f"  GRPO update {grpo_iter+1}/{env.mu}")
            avg_epoch_loss, all_rewards, all_advantages = grpo_update(
                agent,
                rollout_data,
                env,
                optimizer
            )
            # TEST 세션
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
                    action, _ = agent.policy.act(state, deterministic=True)
                    
                    next_state, combined_returns, done, _, action_dict = valid_env.eval_step(action.cpu().numpy(), strategy_returns)
                    next_state = next_state.reshape(1, -1)
                    state = next_state.clone()
                    model_return = pd.concat([model_return, combined_returns], axis=0)

                    if done:
                        break
                
                valid_sharpe = eval_metric.calculate_sharpe_ratio(model_return, risk_free_rate=valid_env.risk_free_rate, annual_factor=valid_env.annual_factor)
                wandb.log({
                    "valid_reward": valid_sharpe,
                })
            agent.policy.train()
            print(f"Train mode status: {agent.policy.training}")  # ✅ True여야 정상
 
                
                
                # optimizer.zero_grad()
                # total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.parameters(), env.clip_grad)
                # optimizer.step()
        wandb.log({
        "train_mean_loss": float(avg_epoch_loss),
        "train_mean_reward": float(torch.tensor(all_rewards).mean()),
        "train_mean_advantage": float(torch.tensor(all_advantages).mean())
    })


        if (global_step + 1) % 2 == 0:
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

    print("\n[GRPO Training Finished]")
    torch.save(agent.policy.state_dict(), env.save_path)

    return agent


def optimize_model_memory(model):
    policy = model.policy

    if hasattr(policy, 'config'):
        policy.config.use_cache = False  # LLM 계열이 아니라면 무시되어도 무방

    if hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()  # FFC/Transformer 계열에서만 유효

    policy = torch.compile(policy)  # PyTorch 2.0+
    model.policy = policy
    return model



def test_grpo(env, model):
    """
    GRPO 모델을 평가하는 함수
    """
    model.policy.eval()
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
        action, _ = model.policy.act(state, deterministic=True)
        next_state, combined_returns, done, _, action_dict = env.eval_step(action.cpu().numpy(), strategy_returns)
        weight_model = pd.concat([weight_model, pd.DataFrame(action_dict).T], axis=0)
        next_state = next_state.reshape(1, -1)
        state = next_state.clone()
        model_return = pd.concat([model_return, combined_returns], axis=0)

        if done:
            break

    return model_return, strategy_returns, weight_model, strategy_weights
