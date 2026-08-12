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

def generate_single_group(idx, model_state, ref_state, env):
    group = []
    state = env.reset()
    state = state.permute(1, 2, 0)
    state = torch.tensor(state, dtype=torch.float32)
    model = PPO(env)
    ref_model = copy.deepcopy(model)
    model.policy_old.to("cpu")
    ref_model.policy_old.to("cpu")

    model.policy_old.load_state_dict(model_state)
    ref_model.policy_old.load_state_dict(ref_state)
    ref_model.policy_old.eval()

    for _ in range(env.num_trajectories):
        trajectory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "ref_log_probs": []}

        with torch.no_grad():
            for _ in range(env.update_interval):
                actions, logprob, _ = model.policy_old.act(state)
                _, ref_logprob, _ = ref_model.policy_old.act(state)

                actions_np = actions.detach().cpu().numpy()
                next_state, reward, done, _ = env.step(actions_np, group_reset=True)

                trajectory["states"].append(state.clone().cpu())
                actions_np = actions_np.reshape(-1)
                trajectory["actions"].append(torch.tensor(actions_np, dtype=torch.float32))
                trajectory["log_probs"].append(logprob.detach())
                trajectory["rewards"].append(torch.tensor(reward, dtype=torch.float32))
                trajectory["ref_log_probs"].append(ref_logprob.detach())

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
    return group


# --- PARALLEL ROLLOUT ENTRY ---
def generate_rollout_data(model, ref_model, env):
    DEBUG_MODE = sys.gettrace() is not None  # VSCode 디버깅 중이면 True

    n_jobs = 1 if DEBUG_MODE else env.num_workers
    print(f"Using {n_jobs} parallel workers for rollout generation.")

    model_state = model.policy_old.state_dict()
    ref_state = ref_model.policy_old.state_dict()
    rollout_groups = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_group)(idx, model_state, ref_state, env)
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
        states = batch["states"].to(env.device, non_blocking=True)
        actions = batch["actions"].to(env.device, non_blocking=True)
        log_probs = batch["log_probs"].to(env.device, non_blocking=True)
        ref_log_probs = batch["ref_log_probs"].to(env.device, non_blocking=True)
        # rewards = batch["rewards"].to(env.device, non_blocking=True)
        advantage = batch["advantage"].to(env.device, non_blocking=True)
        
        logprob_old = log_probs.mean(dim=1).squeeze(-1)
        logprob_ref = ref_log_probs.mean(dim=1).squeeze(-1)

        logprob_new, _ = agent.policy.evaluate(states, actions)
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
        all_rewards.append(batch["rewards"].mean(dim=1).mean())
        all_advantages.append(advantage.mean().item())
    avg_epoch_loss = total_loss / len(loader)

    return avg_epoch_loss, all_rewards, all_advantages



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
                
                # optimizer.zero_grad()
                # total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(agent.parameters(), env.clip_grad)
                # optimizer.step()
                wandb.log({
                "mean_loss": float(avg_epoch_loss),
                "mean_reward": float(torch.tensor(all_rewards).mean()),
                "mean_advantage": float(torch.tensor(all_advantages).mean())
            })

                

                print(f"    → Loss: {float(avg_epoch_loss):.4f}, "
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
