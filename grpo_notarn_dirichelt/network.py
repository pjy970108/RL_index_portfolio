import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical, Dirichlet
# from tarn.TARN import FFCModule
import torch.nn.functional as F  # ✅ 이거 필요!

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

class ActorCritic(nn.Module):
    # def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
    def __init__(self, feature, action_dim, has_continuous_action_space, action_std_init, device, alpha_param):

        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.alpha_param = alpha_param
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            
        # self.ffc = FFCModule(hyperparameters).to(device)  # Use FFC as feature extractor
        self.feature_dim = feature

        # Actor network
        if has_continuous_action_space:
            # self.actor = nn.Sequential(
            #     nn.Linear(self.feature_dim, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, 32),
            #     nn.Tanh(),
            #     nn.Linear(32, action_dim),
            #     # nn.Softplus()  # ensures alpha > 0
            #     # nn.LeakyReLU(),  # 음수 방지용
            #     # nn.Softmax(dim=-1)  # 확률 분포로 변환

            # )

            self.actor = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                # nn.Tanh(),    
                nn.LayerNorm(256),        # ✅ OK during act()
                nn.GELU(),
                nn.Linear(256, 128),
                # nn.Tanh(),    
                nn.LayerNorm(128),        # ✅ OK during act()
                nn.GELU(),
                
                nn.Linear(128, 64),
                # nn.Tanh(),
                nn.LayerNorm(64),        # ✅ OK during act()
                nn.GELU(),

                nn.Linear(64, 32),
                nn.Tanh(),

                nn.Linear(32, action_dim)
)
      

        else:
            self.actor = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
        else:
            print("WARNING: Calling set_action_std() on a discrete action space policy")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    
    
    def act(self, state, deterministic=False):
        # state, attn_scores = self.ffc(state)
        # if self.has_continuous_action_space:
        #     # 연속 액션
        #     action_mean = self.actor(state)
        #     if deterministic:
        #         # print("action_mean", action_mean)
        #         return action_mean.detach(), None  # logprob은 None으로
                
        #     cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(action_mean.device)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        #     action = dist.sample()
        #     action_logprob = dist.log_prob(action)
        #     # return action, action_logprob, self.critic(state)
        #     return action.detach(), action_logprob.detach()
        if self.has_continuous_action_space:
            raw_alpha = self.actor(state)
            alpha = F.softplus(raw_alpha) * self.alpha_param + 1e-3  # 극단적 분포 유도
            dist = Dirichlet(alpha)

            if deterministic:
                print("alpha_sum", alpha.sum(dim=-1, keepdim=True))
                action = alpha / alpha.sum(dim=-1, keepdim=True)  # Expected value
                print("action", action)
                return action.detach(), None  # 확률(weight) 자체를 리턴


            action = dist.sample()
            action_logprob = dist.log_prob(action)

        else:
            # 이산 액션
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            if deterministic:
                # 가장 확률이 높은 행동(=argmax) 선택
                action = torch.argmax(action_probs, dim=-1)
                action_logprob = dist.log_prob(action)
            else:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        return action.detach().squeeze(), action_logprob.detach()



    def evaluate(self, state, action):
        batch_size = state.size(0)
        # state = state.view(-1, state.size(3), state.size(4))
        # state, _ = self.ffc(state)
        state = state.view(batch_size, -1, self.feature_dim)
        # print("state", state)
        # if self.has_continuous_action_space:
        #     raw_alpha = self.actor(state)
        #     alpha = F.softplus(raw_alpha) * self.alpha_param + 1e-3  # 극단적 분포 유도
        #     dist = Dirichlet(alpha)

        # if self.has_continuous_action_space:
        #     action_mean = self.actor(state)
        #     action_var = self.action_var.expand_as(action_mean)
        #     cov_mat = torch.diag_embed(action_var)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        if self.has_continuous_action_space:
            raw_alpha = self.actor(state)
            alpha = F.softplus(raw_alpha) * self.alpha_param + 1e-3
            dist = Dirichlet(alpha)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, dist_entropy
        