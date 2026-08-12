import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random
from collections import deque, namedtuple
try:
    from .network import ActorNetwork, CriticNetwork
except ImportError:
    from network import ActorNetwork, CriticNetwork
import wandb
# from buffer import Replay_Buffer
# #### set device####
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")

# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.stack([torch.tensor(e.state, dtype=torch.float32) for e in experiences]).to(self.device)
        actions = torch.LongTensor([[e.action] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([[e.reward] for e in experiences]).to(self.device)
        next_states = torch.stack([torch.tensor(e.next_state, dtype=torch.float32) for e in experiences]).to(self.device)
        dones = torch.FloatTensor([[e.done] for e in experiences]).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_param, from_param in zip(to_model.parameters(), from_model.parameters()):
        to_param.data.copy_(from_param.data.clone())


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
        - patience (int): 개선이 없더라도 기다릴 에포크 수.
        - min_delta (float): 개선으로 간주되는 최소 변화량.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('-inf')  # 초기값을 -무한대로 설정
        self.early_stop = False
        
    def __call__(self, current_score):
        # 상대적 개선 평가
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # 개선되면 카운터 초기화
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ===== Discrete SAC Agent =====
class DiscreteSACAgent:
    def __init__(self, env):
        self.device = env.device
        self.feature_dim = env.feature_dim
        self.asset_dim = env.asset_dim
        self.state_dim = self.feature_dim * self.asset_dim
        self.action_dim = env.action_dim
        self.gamma = env.gamma
        self.tau = env.tau
        self.alpha = env.alpha_param
        self.lr_actor = env.lr_actor
        self.lr_critic = env.lr_critic
        self.automatic_entropy_tuning  = env.automatically_tune_entropy_hyperparameter
        self.replay_buffer = ReplayBuffer(
            buffer_size=env.buffer_size,
            batch_size=env.mini_batch_size,
            seed=env.seed,
            device=self.device
        )


        if eval==False:
            self.save_path = config.save_path


        self.actor = ActorNetwork(self.state_dim,  env.action_dim).to(env.device)
        self.critic1 = CriticNetwork(self.state_dim,  env.action_dim).to(env.device)
        self.critic2 = CriticNetwork(self.state_dim,  env.action_dim).to(env.device)
        self.critic1_target = CriticNetwork(self.state_dim,  env.action_dim).to(env.device)
        self.critic2_target = CriticNetwork(self.state_dim,  env.action_dim).to(env.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr= self.lr_actor)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr= self.lr_critic)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr= self.lr_critic)

        if self.automatic_entropy_tuning:
            self.target_entropy = -np.log(1.0 / self.action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=env.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_actor)
        else:
            self.log_alpha = None

    def select_action(self, state, deterministic=False):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            return torch.argmax(probs, dim=1)
        else:
            action = dist.sample()

            return action.detach()
        
    def update(self, replay_buffer):
        if len(replay_buffer) < replay_buffer.batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # Next state actions and log probs
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)      # [batch_size, action_dim]
            next_log_probs = next_log_probs.squeeze()
            # dist_next = torch.distributions.Categorical(next_probs)
            # next_log_probs = dist_next.log_prob(torch.arange(self.action_dim).to(self.device)).repeat(next_states.size(0), 1)

            q1_next = self.critic1_target(next_states)
            q2_next = self.critic2_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)
            next_probs = next_probs.squeeze(1)      # [3, 8]
            min_q_next = min_q_next.squeeze(1)      # [3, 8]
            next_value = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            # next_value = next_value.squeeze(1)  # [3, 1]
            q_target = rewards + (1 - dones) * self.gamma * next_value.detach()

        # Current Q estimates
        q1 = self.critic1(states).squeeze(1)  # [3, 8]
        q2 = self.critic2(states).squeeze(1)  # [3, 8]
        actions = actions.view(-1, 1).long()
        q1 = q1.gather(1, actions)  # [3, 1]
        q2 = q2.gather(1, actions)  # [3, 1]
        # q1 = self.critic1(states).gather(1, actions)
        # q2 = self.critic2(states).gather(1, actions)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(torch.arange(self.action_dim).to(self.device))
        q1_pi = self.critic1(states).squeeze(1)
        q2_pi = self.critic2(states).squeeze(1)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_probs * probs, dim=1)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy temperature tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs.sum(dim=1) + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            
            wandb.log({
                "critic1_loss": critic1_loss.item(),
                "critic2_loss": critic2_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alpha
            })
        else:
            wandb.log({
                "critic1_loss": critic1_loss.item(),
                "critic2_loss": critic2_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha": self.alpha
            })


        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
        }, path)

    def load(self, path):
        # checkpoint = torch.load(path, map_location=self.device)
        # self.actor.load_state_dict(checkpoint['actor'])
        # self.critic1.load_state_dict(checkpoint['critic1'])
        # self.critic2.load_state_dict(checkpoint['critic2'])
        # self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        # self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        # self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        # self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        # self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        # if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
        #     self.log_alpha = checkpoint['log_alpha']
        #     self.alpha = self.log_alpha.exp().item()
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint)
        print(path + " loaded.")


# class SAC_Discrete:
#     def __init__(self, env):
#         """
#         state_dim, action_dim: 상태 및 행동 공간의 차원 수
#         lr_actor, lr_critic: 정책 네트워크와 가치 네트워크의 학습률
#         gamma: 할인 계수
#         K_epochs: 정책을 업데이트할 에포크 수
#         eps_clip: 정책 업데이트 시 클리핑 범위 (PPO의 핵심)
#         has_continuous_action_space: 연속적 행동 공간 여부
#         action_std_init: 연속적 행동 공간에서 행동의 표준 편차 초기값
#         """
        
#         self.has_continuous_action_space = env.has_continuous_action_space
#         self.gamma = env.gamma
#         self.epsilon = env.epsilon
#         self.mu = env.mu
#         # self.state_dim = config.state_dim
#         self.asset_dim = env.asset_dim
#         self.action_dim = env.action_dim
#         self.device = env.device
#         self.lr_actor = env.lr_actor
#         self.lr_critic = env.lr_critic
#         if eval==False:
#             self.save_path = config.save_path
#         self.feature_dim = env.feature_dim
#         self.action_std_init = env.action_std_init
#         # Initialize buffer
#         self.buffer = RolloutBuffer()

#         # Actor, Critic 네트워크 정의함. 상태 및 행동 차원, 공간유형을 받아 초기화함.
#         # self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
#         self.A2C = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device).to(self.device)

#         self.critic_local = self.A2C.critic_1
#         self.critic_local_2 = self.A2C.critic_2
        
#         self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
#                                                  lr=self.lr_critic, eps=1e-4)
#         self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
#                                                    lr=self.lr_critic, eps=1e-4)
        
#         self.critic_target = copy.deepcopy(self.critic_local)
#         self.critic_target_2 = copy.deepcopy(self.critic_local_2)
        
#         copy_model_over(self.critic_local, self.critic_target)
#         copy_model_over(self.critic_local_2, self.critic_target_2)
        
#         self.buffer = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
#                                     env.seed)
        
#         self.actor_local =self.A2C.actor

#         self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
#                                           lr=self.lr_actor, eps=1e-4)
        
        
#         self.automatic_entropy_tuning  = env.automatically_tune_entropy_hyperparameter
        
#         if self.automatic_entropy_tuning:
#             self.target_entropy = -torch.prod(torch.Tensor(env.action_dim).to(self.device)).item() # heuristic value from the paper
#             self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
#             self.alpha = self.log_alpha.exp()
#             self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr_actor, eps=1e-4)
#         else:
#             self.alpha = env.entropy_term_weight

#         self.add_extra_noise = env.add_extra_noise
        
#         if self.add_extra_noise:
#             self.noise = OU_Noise(self.action_dim, env.seed, 0,
#                                   0.15, 0.25)

#         self.do_evaluation_iterations = env.do_evaluation_iterations


#         # self.optimizer = torch.optim.Adam([
#         #     {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
#         #     {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
#         # ])

#         # # 이전 정책 네트워크 정의함. 현재 정책 네트워크의 가중치를 복사하여 초기화함.
#         # # self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
#         # self.policy_old = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device).to(self.device)

#         # # policy의 가중치를 policy_old에 복사함
#         # self.policy_old.load_state_dict(self.policy.state_dict())

#         # # Loss function
#         # self.MseLoss = nn.MSELoss()
#         # # self.huber_loss = nn.SmoothL1Loss()


#     def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
#         """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
#          term is taken into account"""
#         with torch.no_grad():
#             next_state_action, (action_probabilities, log_action_probabilities), _ = self.actor_local.act(next_state_batch)
#             qf1_next_target = self.critic_target(next_state_batch)
#             qf2_next_target = self.critic_target_2(next_state_batch)
#             min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
#             min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
#             next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

#         qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
#         qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
#         qf1_loss = F.mse_loss(qf1, next_q_value)
#         qf2_loss = F.mse_loss(qf2, next_q_value)
#         return qf1_loss, qf2_loss

#     def calculate_actor_loss(self, state_batch):
#         """Calculates the loss for the actor. This loss includes the additional entropy term"""
#         action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
#         qf1_pi = self.critic_local(state_batch)
#         qf2_pi = self.critic_local_2(state_batch)
#         min_qf_pi = torch.min(qf1_pi, qf2_pi)
#         inside_term = self.alpha * log_action_probabilities - min_qf_pi
#         policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
#         log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
#         return policy_loss, log_action_probabilities





#     def produce_action_and_action_info(self, state):
#         """Given the state, produces an action, the probability of the action, the log probability of the action, and
#         the argmax action"""
#         action_probabilities = self.actor_local(state)
#         max_probability_action = torch.argmax(action_probabilities, dim=-1)
#         action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
#         action = action_distribution.sample().cpu()
#         # Have to deal with situation of 0.0 probabilities because we can't do log 0

#         log_action_probabilities = torch.log(action_probabilities + z)
#         return action, (action_probabilities, log_action_probabilities), max_probability_action





#     def select_action(self, state, deterministic=False):
#         # 만약 연속 행동 공간이면
#         with torch.no_grad():
#             state = state.to(self.device, dtype=torch.float)
#             action, action_probs, action_logprob, max_probability_action = self.actor_local.act(state, deterministic)

#         if not deterministic:  # 학습용 데이터는 stochastic일 때만 저장
#             self.buffer.states.append(state)
#             self.buffer.actions.append(action)
#             self.buffer.logprobs.append(action_logprob)
#             self.buffer.state_values.append(state_val)

#         return action.detach().cpu().numpy().flatten()

#     # def calculate_actor_loss(self, states):
        



#     # 정책 업데이트 
#     def update(self, mini_batch_size):
#         # Monte Carlo 방식으로 리턴 계산
#         rewards = []
#         discounted_reward = 0
#         # buffer rewards가 들어감
#         # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#         #     # 반약 is_terminal이 True면 discounted reward를 0으로 설정함.
#         #     if is_terminal:
#         #         discounted_reward = 0
            
#         #     # 계산된 리턴을 텐서로 변환하고 보상을 정규화함. 
#         #     discounted_reward = reward + (self.gamma * discounted_reward)
#         #     rewards.insert(0, discounted_reward)
#         for i in reversed(range(len(self.buffer.rewards))):
#             if self.buffer.is_terminals[i]:
#                 discounted_reward = self.buffer.state_values[i]  # V(s_T) 추가 고려
#             discounted_reward = self.buffer.rewards[i] + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#         # Normalizing the rewards, 성능 향상을 위해 필요함.
#         # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
#         # rewards_norm = (torch.stack(rewards) - torch.stack(rewards).mean()) / (torch.stack(rewards).std() + 1e-7)
#         # rewards_norm = rewards_norm.squeeze()  # → shape: [120]
#         # print("rewards_norm mean/std:", rewards_norm.mean(), rewards_norm.std())
#         returns = torch.stack(rewards).detach()  # critic target
#         returns = returns.squeeze()

#         # # 표준편차가 0인지 확인
#         # if torch.isnan(rewards.std()) or rewards.std() == 0:
#         #     rewards = rewards  # 정규화하지 않고 그대로 사용
#         # else:
#         #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#         # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#         # rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

#         # convert list to tensor
#         # 버퍼의 데이터 변환 및 분리 - 버퍼에 저장된 상태, 행동, 로그 확률, 상태 가치를 텐서로 변환하고 detach를 사용해 경사 계산에서 제외함.
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0))
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0))
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
#         old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
#         # calculate advantages 정규화된 리턴에서 상태 가치를 뺀값임.
#         # 각상태에서 행동이 얼마나 좋은지 나타냄.
#         # advantages = rewards - old_state_values # 200일간의 advantage 계산
#         advantages = (returns - old_state_values).detach()
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)  # 정규화 ✅
        
#         dataset_size = old_states.size(0)
#         mini_batch_size = mini_batch_size or dataset_size
#         # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
#         # Optimize policy for K epochs
#         # K_epoch 동안 정책을 반복적으로 업데이트함.
#         # total_policy_loss = 0
#         # total_value_loss = 0
#         # total_entropy_bonus = 0
#         total_policy_loss = 0
#         total_value_loss = 0
#         total_entropy_bonus = 0
#         for _ in range(self.mu):
#             indices = torch.randperm(dataset_size)
#             for start in range(0, dataset_size, mini_batch_size):
#                 end = start + mini_batch_size
#                 mb_idx = indices[start:end]

#                 mb_states = old_states[mb_idx]
#                 mb_actions = old_actions[mb_idx]
#                 mb_logprobs = old_logprobs[mb_idx]
#                 mb_rewards = returns[mb_idx]
#                 mb_advantages = advantages[mb_idx]

#                 logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
#                 # print("=== Critic Output Check ===")
#                 # print("state_values (raw):", state_values[:5].view(-1))  # 5개만
#                 # print("state_values mean/std:", state_values.mean().item(), state_values.std().item())
#                 # print("===========================")
#                 state_values = torch.squeeze(state_values)
#                 ratios = torch.exp(logprobs - mb_logprobs)
#                 surr1 = ratios * mb_advantages
#                 surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages

#                 policy_loss = -torch.min(surr1, surr2).mean()
#                 value_loss = 0.5 * self.MseLoss(state_values, mb_rewards)
#                 entropy_bonus = dist_entropy.mean()

#                 loss = policy_loss + value_loss - self.entropy_weight * entropy_bonus

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#                 total_policy_loss += policy_loss.item()
#                 total_value_loss += value_loss.item()
#                 total_entropy_bonus += entropy_bonus.item()
#         avg_policy_loss = total_policy_loss / (self.mu * (dataset_size // mini_batch_size))
#         avg_value_loss = total_value_loss / (self.mu * (dataset_size // mini_batch_size))
#         avg_entropy_bonus = total_entropy_bonus / (self.mu * (dataset_size // mini_batch_size))
#         avg_total_loss = avg_policy_loss + avg_value_loss - self.entropy_weight * avg_entropy_bonus
#         self.last_loss = avg_total_loss
#         total_raw_reward = sum(rewards)

#         self.policy_old.load_state_dict(self.policy.state_dict())
#         self.buffer.clear()

#         return avg_policy_loss, avg_value_loss, avg_entropy_bonus, avg_total_loss

            

#     def save(self):
#         # 모델 저장
#         torch.save(self.policy_old.state_dict(), self.save_path)
   
#     def load(self, model_path):
#         # 모델 불러오기
#         self.policy_old.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
#         self.policy.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
#         print(model_path + " loaded.")
        
        
        
#     def get_last_metrics(self):
#         """
#         마지막 업데이트 단계에서 기록한 메트릭들을 반환합니다.
#         """
#         if self.last_loss is None:
#             raise ValueError("No metrics available. Make sure to run update() first.")
        
#         return {
#             "loss": self.last_loss
#         }



if __name__ == "__main__":
    import torch
    import copy
    import pandas as pd
    from enviroment import EnvConfig, Stock_Env
    import numpy as np
    from agent import PPO
    train_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    train_dataset = pd.read_csv("/Users/pjy97/Desktop/hyu/research/RL/code/data/train_data.csv", index_col=0)
    test_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    config = EnvConfig(train_tensor, train_dataset, 22)
    
    env = copy.deepcopy(Stock_Env(train_tensor, train_dataset, 22))
    
    agent = PPO(config)