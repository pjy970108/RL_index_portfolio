import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical, Dirichlet
from TARN_PPO.TARN import FFCModule

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

class ActorCritic(nn.Module):
    # def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
    def __init__(self, feature, action_dim, has_continuous_action_space, action_std_init, hyperparameters, device):

        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            
        self.ffc = FFCModule(hyperparameters).to(device)  # Use FFC as feature extractor
        self.feature_dim = hyperparameters["final_conv_config"]["out_channels"] * feature

        # Actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Softplus()  # ensures alpha > 0
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
        
        # Critic network
        # self.critic = nn.Sequential(
        #     nn.Linear(self.feature_dim, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 1)
        # )
        
        # Critic network
        # self.critic = nn.Sequential(
        #     nn.Linear(self.feature_dim, 32),
        #     nn.LeakyReLU(0.2),  # 변경
        #     nn.Linear(32, 32),
        #     nn.LeakyReLU(0.2),  # 변경
        #     nn.Linear(32, 1)
        # )
        # self.critic = nn.Sequential(
        #     nn.Linear(self.feature_dim, 128),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(0.2),

        #     nn.Linear(32, 1)
        # )

        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
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
    
    

    # def act(self, state):
    #     state = self.ffc(state)

    #     if self.has_continuous_action_space:
    #         action_mean = self.actor(state)
    #         cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    #         dist = MultivariateNormal(action_mean, cov_mat)
    #     else:
    #         action_probs = self.actor(state)
    #         dist = Categorical(action_probs)
    #     print("action_mean: ", action_mean)
    #     action = dist.sample()
    #     action_logprob = dist.log_prob(action)
    #     state_val = self.critic(state)
    #     return action.detach(), action_logprob.detach(), state_val.detach()
    
    
    def act(self, state, deterministic=False):
        state, attn_scores = self.ffc(state)
        # print("state", state)

        # if self.has_continuous_action_space:
        #     # 연속 액션
        #     action_mean = self.actor(state)
        #     if deterministic:
        #         # print("action_mean", action_mean)
        #         return action_mean.detach(), None, self.critic(state).detach(), attn_scores  # logprob은 None으로
                
        #     cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        #     dist = MultivariateNormal(action_mean, cov_mat)
        #     action = dist.sample()
        #     action_logprob = dist.log_prob(action)
            # return action, action_logprob, self.critic(state)
        if self.has_continuous_action_space:
            alpha = self.actor(state) + 1e-3  # ensure strictly > 0
            dist = Dirichlet(alpha)

            if deterministic:
                action = alpha / alpha.sum(dim=-1, keepdim=True)  # expected value of Dirichlet
                return action.detach(), None, self.critic(state).detach(), attn_scores

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

        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach(),attn_scores



    def evaluate(self, state, action):
        state = state.view(-1, state.size(2), state.size(3))
        state, _ = self.ffc(state)
        state = state.view(-1, self.feature_dim)
        # print("state", state)
        if self.has_continuous_action_space:
            # action_mean = self.actor(state)
            # action_var = self.action_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var)
            # dist = MultivariateNormal(action_mean, cov_mat)
            # check_nan_1 = torch.isnan(action_mean).any()
            # check_nan_2 = torch.isnan(action_var).any()
            # check_nan_3 = torch.isnan(cov_mat).any()
            # # check_nan_4 = torch.isnan(dist).any()
            # if check_nan_1 or check_nan_2 or check_nan_3:
            #     print("NAN detected in actor")
            # `state`와 `action_mean`의 값을 출력
            alpha = self.actor(state) + 1e-3
            dist = Dirichlet(alpha)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

        