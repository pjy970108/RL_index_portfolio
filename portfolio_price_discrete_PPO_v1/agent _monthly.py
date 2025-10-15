import torch
import torch.nn as nn
from network import ActorCritic
import wandb
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

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
    
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



class PPO:
    def __init__(self, env):
        """
        state_dim, action_dim: 상태 및 행동 공간의 차원 수
        lr_actor, lr_critic: 정책 네트워크와 가치 네트워크의 학습률
        gamma: 할인 계수
        K_epochs: 정책을 업데이트할 에포크 수
        eps_clip: 정책 업데이트 시 클리핑 범위 (PPO의 핵심)
        has_continuous_action_space: 연속적 행동 공간 여부
        action_std_init: 연속적 행동 공간에서 행동의 표준 편차 초기값
        """
        
        self.has_continuous_action_space = env.has_continuous_action_space
        self.gamma = env.gamma
        self.epsilon = env.epsilon
        self.mu = env.mu
        # self.state_dim = config.state_dim
        self.asset_dim = env.asset_dim
        self.action_dim = env.action_dim
        self.action_std_init = env.action_std_init
        self.device = env.device
        self.lr_actor = env.lr_actor
        self.lr_critic = env.lr_critic
        if eval==False:
            self.save_path = config.save_path

        self.entropy_weight = env.entropy_weight
        self.feature_dim = env.feature_dim

        # Initialize buffer
        self.buffer = RolloutBuffer()

        # Actor, Critic 네트워크 정의함. 상태 및 행동 차원, 공간유형을 받아 초기화함.
        # self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
        self.policy = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        # 이전 정책 네트워크 정의함. 현재 정책 네트워크의 가중치를 복사하여 초기화함.
        # self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
        self.policy_old = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device).to(self.device)

        # policy의 가중치를 policy_old에 복사함
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Loss function
        self.MseLoss = nn.MSELoss()
        # self.huber_loss = nn.SmoothL1Loss()



    # 연속된 공간에서 행동의 표준 편차 설정함
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std_init = new_action_std

            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        # else:
        #     print("--------------------------------------------------------------------------------------------")
        #     print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
       #     print("--------------------------------------------------------------------------------------------")

    # 조건을 만족할때 행동의 표준 편차 감소시킴
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std_init = self.action_std_init - action_std_decay_rate
            self.action_std_init = round(self.action_std_init, 4)
            if (self.action_std_init <= min_action_std):
                self.action_std_init = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std_init)
            else:
                print("setting actor output action_std to : ", self.action_std_init)
            self.set_action_std(self.action_std_init)

        # else:
        #     print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        # print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, deterministic=False):
        # 만약 연속 행동 공간이면
        with torch.no_grad():
            state = state.to(self.device, dtype=torch.float)
            action, action_logprob, state_val = self.policy_old.act(state, deterministic)

        if not deterministic:  # 학습용 데이터는 stochastic일 때만 저장
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()




    # 정책 업데이트 
    def update(self, mini_batch_size):
        # Monte Carlo 방식으로 리턴 계산
        rewards = []
        discounted_reward = 0
        # buffer rewards가 들어감
        # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        #     # 반약 is_terminal이 True면 discounted reward를 0으로 설정함.
        #     if is_terminal:
        #         discounted_reward = 0
            
        #     # 계산된 리턴을 텐서로 변환하고 보상을 정규화함. 
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        for i in reversed(range(len(self.buffer.rewards))):
            if self.buffer.is_terminals[i]:
                discounted_reward = self.buffer.state_values[i]  # V(s_T) 추가 고려
            discounted_reward = self.buffer.rewards[i] + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards, 성능 향상을 위해 필요함.
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # rewards_norm = (torch.stack(rewards) - torch.stack(rewards).mean()) / (torch.stack(rewards).std() + 1e-7)
        # rewards_norm = rewards_norm.squeeze()  # → shape: [120]
        # print("rewards_norm mean/std:", rewards_norm.mean(), rewards_norm.std())
        returns = torch.stack(rewards).detach()  # critic target
        returns = returns.squeeze()

        # # 표준편차가 0인지 확인
        # if torch.isnan(rewards.std()) or rewards.std() == 0:
        #     rewards = rewards  # 정규화하지 않고 그대로 사용
        # else:
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # 버퍼의 데이터 변환 및 분리 - 버퍼에 저장된 상태, 행동, 로그 확률, 상태 가치를 텐서로 변환하고 detach를 사용해 경사 계산에서 제외함.
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0))
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0))
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
        # calculate advantages 정규화된 리턴에서 상태 가치를 뺀값임.
        # 각상태에서 행동이 얼마나 좋은지 나타냄.
        # advantages = rewards - old_state_values # 200일간의 advantage 계산
        advantages = (returns - old_state_values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)  # 정규화 ✅
        
        dataset_size = old_states.size(0)
        mini_batch_size = mini_batch_size or dataset_size
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        # Optimize policy for K epochs
        # K_epoch 동안 정책을 반복적으로 업데이트함.
        # total_policy_loss = 0
        # total_value_loss = 0
        # total_entropy_bonus = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_bonus = 0
        for _ in range(self.mu):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_states = old_states[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_rewards = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                # print("=== Critic Output Check ===")
                # print("state_values (raw):", state_values[:5].view(-1))  # 5개만
                # print("state_values mean/std:", state_values.mean().item(), state_values.std().item())
                # print("===========================")
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - mb_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * self.MseLoss(state_values, mb_rewards)
                entropy_bonus = dist_entropy.mean()

                loss = policy_loss + value_loss - self.entropy_weight * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_bonus += entropy_bonus.item()
        avg_policy_loss = total_policy_loss / (self.mu * (dataset_size // mini_batch_size))
        avg_value_loss = total_value_loss / (self.mu * (dataset_size // mini_batch_size))
        avg_entropy_bonus = total_entropy_bonus / (self.mu * (dataset_size // mini_batch_size))
        avg_total_loss = avg_policy_loss + avg_value_loss - self.entropy_weight * avg_entropy_bonus
        self.last_loss = avg_total_loss
        total_raw_reward = sum(rewards)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return avg_policy_loss, avg_value_loss, avg_entropy_bonus, avg_total_loss

            

    def save(self):
        # 모델 저장
        torch.save(self.policy_old.state_dict(), self.save_path)
   
    def load(self, model_path):
        # 모델 불러오기
        self.policy_old.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        print(model_path + " loaded.")
        
        
        
    def get_last_metrics(self):
        """
        마지막 업데이트 단계에서 기록한 메트릭들을 반환합니다.
        """
        if self.last_loss is None:
            raise ValueError("No metrics available. Make sure to run update() first.")
        
        return {
            "loss": self.last_loss
        }



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