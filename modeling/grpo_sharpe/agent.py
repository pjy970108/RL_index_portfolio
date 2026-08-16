import torch
import torch.nn as nn

import sys
import os

try:
    from network import ActorCritic
    from enviroment import Stock_Env
except:
    from .network import ActorCritic
    from .enviroment import Stock_Env

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
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    
class PPO:
    def __init__(self, config, eval=False):
        """
        state_dim, action_dim: 상태 및 행동 공간의 차원 수
        lr_actor, lr_critic: 정책 네트워크와 가치 네트워크의 학습률
        gamma: 할인 계수
        K_epochs: 정책을 업데이트할 에포크 수
        eps_clip: 정책 업데이트 시 클리핑 범위 (PPO의 핵심)
        has_continuous_action_space: 연속적 행동 공간 여부
        action_std_init: 연속적 행동 공간에서 행동의 표준 편차 초기값
        """
        
        self.has_continuous_action_space = config.has_continuous_action_space
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.asset_dim = config.asset_dim
        self.feature_dim = config.feature_dim
        self.action_dim = config.action_dim
        self.action_std_init = config.action_std_init
        self.device = config.device
        self.lr_actor = config.lr_actor
        if eval==False:
            self.save_path = config.save_path
        self.entropy_weight = config.entropy_weight
        self.use_cache = True
        self.alpha_param = config.alpha_param

        # Initialize buffer
        self.buffer = RolloutBuffer()

        # Actor, Critic 네트워크 정의함. 상태 및 행동 차원, 공간유형을 받아 초기화함.
        # self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
        self.policy = ActorCritic(self.asset_dim * self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device, self.alpha_param).to(self.device)

        # self.optimizer = torch.optim.Adam([
        #     # {'params': self.policy.ffc.parameters(), 'lr': self.lr_actor},  # FFC 학습 포함!
        #     {'params': self.policy.actor.parameters(), 'lr': self.lr_actor}
        # ])

        # 이전 정책 네트워크 정의함. 현재 정책 네트워크의 가중치를 복사하여 초기화함.
        # self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
        self.policy_old = ActorCritic(self.asset_dim * self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.device, self.alpha_param).to(self.device)

        # policy의 가중치를 policy_old에 복사함
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Loss function
        # self.MseLoss = nn.MSELoss()
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
        if self.has_continuous_action_space== True:
            with torch.no_grad():
                state = state.to(self.device, dtype=torch.float)
                action, action_logprob = self.policy_old.act(state, deterministic)
 
        else:
            state = state.to(self.device, dtype=torch.float)
            action, action_logprob = self.policy_old.act(state)
 
        if not deterministic:  # 학습용 데이터는 stochastic일 때만 저장
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten(), action_logprob

    def save(self):
        torch.save(self.policy_old.state_dict(), self.save_path)
        
    def load(self, model_path):

        raw_state = torch.load(model_path, map_location=self.device)

        # _orig_mod. prefix 제거
        if any(k.startswith('_orig_mod.') for k in raw_state.keys()):
            raw_state = {k.replace('_orig_mod.', ''): v for k, v in raw_state.items()}

        self.policy_old.load_state_dict(raw_state)
        self.policy.load_state_dict(raw_state)
        print(model_path + " loaded.")


        # self.policy_old.load_state_dict(torch.load(self.save_path, map_location=self.device))
        # self.policy.load_state_dict(torch.load(self.save_path, map_location=self.device))
    
    def parameters(self):
        return self.policy.parameters()
    

class ForwardableActorCritic(nn.Module):
    def __init__(self, policy_model):
        super().__init__()
        self.policy = policy_model

    def forward(self, x):
        # x: (C, H, W) → 모델 내부는 (B, C, H, W) 필요할 수 있음
        # if x.dim() == 3:
        #     x = x.unsqueeze(0)  # batch dimension 추가
        x = x.squeeze(0)  # batch dimension 제거
        # forward를 대신 act()로 호출해서 FLOPs 측정 가능하게 처리
        out, _ = self.policy.act(x)
        return out
