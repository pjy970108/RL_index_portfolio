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

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, norm_rewards):
        self.trajectories = trajectories
        self.norm_rewards = norm_rewards

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return {
            "states": traj["states"],
            "actions": traj["actions"],
            "log_probs": traj["log_probs"],
            "ref_log_probs": traj["ref_log_probs"],
            "rewards": traj["rewards"],
            "advantage": self.norm_rewards[idx],
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

def generate_rollout_data(model, ref_model, env):
    """
    주식 환경 기반으로 GRPO 스타일의 rollout group 생성
    - 각 group은 같은 초기 상태에서 num_generations 번 rollout
    - 각 rollout은 trajectory로 저장됨
    """    
    roll_out_groups = []
    # Random한 환경에서 실행한 Trajectory
    # 다른 초기 상태
    # 그룹별, trajectory states저장
    # 그룹별로 다른 초기 상태에서 시작
    for idx in range(env.batch_samples):
        group = []
        state = env.reset()
        state = state.permute(1, 2, 0)
        state = torch.tensor(state, dtype=torch.float32)

        # 그룹별로 몇번 trajectory를 만들것인지
        # gen_idx만큼의  Trajectory를 만듬
        # 같은 환경에서 Num_trajectories 만큼 만듦
        for gen_idx in range(env.num_trajectories):
            trajectory = {
                "states": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "ref_log_probs": []
            }
            # curr_state = state.clone()

            with torch.no_grad():

                # done이 True가 될 때까지 반복 3, 6, 12 개월 마다 설정된 값만큼 돈다
                for day_step in range(env.update_interval):
                    # 정책만 가져온다.
                    actions, action_logprob, att_score = model.policy_old.act(state)
                    _, ref_log_probs, _ = ref_model.policy_old.act(state)

                    actions = actions.detach().cpu().numpy()
                    next_state, rewards, done, reward_dict = env.step(actions, group_reset = True)
                    rewards = torch.tensor(rewards, dtype=torch.float32).cpu()
                    
                    trajectory["states"].append(state.clone().cpu())
                    actions = actions.reshape(-1)
                    trajectory["actions"].append(torch.tensor(actions, dtype=torch.float32).cpu())
                    trajectory["log_probs"].append(action_logprob.detach().cpu())
                    trajectory["rewards"].append(rewards)
                    trajectory["ref_log_probs"].append(ref_log_probs.detach().cpu())
                    next_state = next_state.permute(1, 2, 0)
                    state = next_state.clone()

                    if done:
                        print(env.idx)
                        state = env.states[env.idx]
                        state = state.permute(1, 2, 0)
                        state= state.clone()
                        pass

            for key in trajectory:
                trajectory[key] = torch.stack(trajectory[key])
            group.append(trajectory)
        roll_out_groups.append(group)
    return roll_out_groups


# def grpo_update(agent, rollout_groups, env):
#     """
#     GRPO loss 계산 및 업데이트 수행
#     rollout_groups: [ [gen_1, ..., gen_N], [gen_1, ..., gen_N], ... ]
#     """
#     all_trajectories = []
#     norm_rewards_list = []

#     # 그룹 단위로 reward 정규화 수행 후 전체 batch로 합치기
#     for group in rollout_groups:
#         group_returns = []
#         for traj in group:
#             traj_reward = traj["rewards"].sum()  # trajectory 전체에 대한 reward 합산
#             group_returns.append(traj_reward)
#         group_returns = torch.stack(group_returns)  # tensor로 변환

#         mean_r = group_returns.mean()
#         std_r = group_returns.std(unbiased=False) + 1e-8
#         norm_rewards = (group_returns - mean_r) / std_r
#         # 전체 group 통합합
#         all_trajectories.extend(group)
#         # 정규화된 reward를 리스트에 추가
#         norm_rewards_list.extend(norm_rewards)

#     all_losses = []
#     all_advantages = []
#     all_rewards = []
#     total_loss = 0
#     # 랜덤하게 trajectory를 선택해 batch 학습
#     max_batch = env.mini_batch_size
#     # 모든 group의 모든 Trajectory(5*20) 데이터와 mini_batch_size 중 작은 값으로 샘플링
#     batch_indices = random.sample(range(len(all_trajectories)), k=min(max_batch, len(all_trajectories)))

#     for i in batch_indices:
#         traj = all_trajectories[i]
#         traj["actions"] = traj["actions"].to(env.device)
#         traj["states"] = traj["states"].to(env.device)
#         traj["log_probs"] = traj["log_probs"].to(env.device)
#         traj["ref_log_probs"] = traj["ref_log_probs"].to(env.device)
#         traj["rewards"] = traj["rewards"].to(env.device)

#         logprob_old = traj["log_probs"].mean()
#         logprob_ref = traj["ref_log_probs"].mean()

#         logprob_new, _ = agent.policy.evaluate(traj["states"], traj["actions"])
#         logprob_new = logprob_new.mean()

#         ratio = torch.exp(logprob_new - logprob_old)
#         clipped_ratio = torch.clamp(ratio, 1 - env.epsilon, 1 + env.epsilon)

#         advantage = norm_rewards_list[i]
#         surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

#         kl = torch.exp(logprob_ref - logprob_new) - (logprob_ref - logprob_new) - 1
#         loss = -surrogate + env.beta * kl

#         total_loss += loss
#         all_losses.append(loss)
#         all_advantages.append(advantage)
#         all_rewards.append(traj["rewards"].sum())
#     return total_loss, all_rewards, all_advantages


def grpo_update(agent, rollout_groups, env, optimizer):
    """
    전체 trajectory를 DataLoader로 미니배치 학습하고, 
    optimizer는 각 배치마다 step을 진행
    """
    all_trajectories = []
    norm_rewards_list = []

    # reward 정규화
    for group in rollout_groups:
        group_returns = torch.stack([traj["rewards"].sum() for traj in group])
        mean_r = group_returns.mean()
        std_r = group_returns.std(unbiased=False) + 1e-8
        norm_rewards = (group_returns - mean_r) / std_r

        all_trajectories.extend(group)
        norm_rewards_list.extend(norm_rewards)

    dataset = TrajectoryDataset(all_trajectories, norm_rewards_list)
    loader = DataLoader(dataset, batch_size=env.mini_batch_size, shuffle=True)

    total_loss = 0
    all_rewards = []
    all_advantages = []

    for batch in loader:
        batch = {k: v.to(env.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logprob_old = batch["log_probs"].mean(dim=1).squeeze(-1)
        logprob_ref = batch["ref_log_probs"].mean(dim=1).squeeze(-1)
        advantage = batch["advantage"]

        logprob_new, _ = agent.policy.evaluate(batch["states"], batch["actions"])
        logprob_new = logprob_new.mean(dim=1)

        ratio = torch.exp(logprob_new - logprob_old)
        clipped_ratio = torch.clamp(ratio, 1 - env.epsilon, 1 + env.epsilon)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        kl = torch.exp(logprob_ref - logprob_new) - (logprob_ref - logprob_new) - 1
        loss_all = -surrogate + env.beta * kl
        loss = loss_all.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), env.clip_grad)
        optimizer.step()

        total_loss += loss.item()
        all_rewards.append(batch["rewards"].sum().item())
        all_advantages.append(advantage.sum().item())

    return total_loss, all_rewards, all_advantages



def train_with_grpo(agent, env):
    
    best_avg_reward = -float("inf")
    patience_counter = 0
    
    for iteration in range(env.num_iterations):
        print(f"\n[Iteration {iteration+1}/{env.num_iterations}]")
        # 정책 모델 초기화 및 복사사
        ref_model = copy.deepcopy(agent)
        ref_model.policy.eval()
        
        for param in ref_model.policy.parameters():
            param.requires_grad = False
        print("Reference model created")

        # re-initialize optimizer 
        # optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr_actor)
        # agent 학습
        optimizer = torch.optim.Adam(agent.parameters(), lr=env.lr_actor)

        # num_steps만큼 반복
        # batch samples 
        # sample g를 뽑음 
        for step in range(env.num_steps): 
            print(f"\nStep {step+1}/{env.num_steps}")
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    agent,
                    ref_model,
                    env
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            for grpo_iter in range(env.mu):
                print(f"  GRPO update {grpo_iter+1}/{env.mu}")
                total_loss, all_rewards, all_advantages = grpo_update(
                    agent,
                    rollout_data,
                    env,
                    optimizer
                )
                
                # optimizer.zero_grad()
                # total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.parameters(), env.clip_grad)
                # optimizer.step()
                wandb.log({
                "mean_loss": float(total_loss),
                "mean_reward": float(torch.tensor(all_rewards).mean()),
                "mean_advantage": float(torch.tensor(all_advantages).mean())
            })

                

                print(f"    → Loss: {float(total_loss):.4f}, "
                      f"Reward: {float(torch.tensor(all_rewards).mean()):.4f}, "
                      f"Advantage: {float(torch.tensor(all_advantages).mean()):.4f}")
                if (iteration + 1) % 2 == 0 and (grpo_iter + 1) % env.mu == 0:
                    torch.save(agent.policy.state_dict(), env.save_path)
                    wandb.log({"model_saved_at": env.save_path})
                    print(f"💾 Saved model at: {env.save_path}")

                
        # Early stopping 체크
        avg_reward = float(torch.tensor(all_rewards).mean())
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
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



# def setup_training_environment(device_name, dtype = "bfloat16"):
#     """set up mixed precision (for memory optimization)"""
    
#     # Set up random seed
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
    
#     # Set up mixed precision
#     device_type = 'cuda' if 'cuda' in device_name else 'cpu'
#     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
#     ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
#     return {
#         'device': device_name,
#         'ctx': ctx,
#         'device_type': device_type,
#     }
