import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "base_line_model/Task_1_FinRL_DeepSeek_Stock")))
print("Current working directory:", os.getcwd())
print(sys.path)
import dynamic_portfolio as dp
import TARN_PPO.backtesting as backtest
import torch
import copy
import random
import pandas as pd


class EnvConfig:
    def __init__(self, states, real_df, real_numpy_data, hyperparameters, trajectory_length = "1y"):
        self.env_name = "Portfolio Allocation"
        # self.window_size = 22
        # self.rolling_step = 0

        # self.states = states[str(self.rolling_step)] # 한달 기준으로 자르는 코드
        # self.real_data = real_data[self.rolling_step]
        self.states = states # 한달 기준으로 자르는 코드
        self.real_data = real_df
        self.real_numpy_data = real_numpy_data
        self.invest_money = 10000000
        self.n_weight_money = 10000000

        self.epochs = 1
        self.log_interval = 10
        self.update_interval = 10 # 10개의 샘플로 비교함
        self.asset_weight_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        self.reward_cond = "combined"
        self.dynamic_dict = {
            "GTAA": dp.cal_gtaa, 
            "DM": dp.cal_dm,  
            "PAA": dp.cal_paa, 
            "DAA": dp.cal_daa,
            "Sentiment": dp.cal_sentiment,
            "risk": dp.cal_risk}
        
        if trajectory_length == "1y":
            self.update_interval = 252  # 1년 기준 (거래일)
            self.num_trajectories = 5
            self.mini_batch_size = 256
        elif trajectory_length == "6m":
            self.update_interval = 120  # 6개월 기준
            self.num_trajectories = 11
            self.mini_batch_size = 256

        elif trajectory_length == "3m":
            self.update_interval = 60   # 3개월 기준
            self.num_trajectories = 23
            self.mini_batch_size = 256

            
        self.num_iterations= 2 # 평가 모델의 정책을 몇번마다 만들것인가
        self.num_steps = 50 # max step을 몇번 할것인가
        self.num_group = 3
        self.mu = 3
        self.clip_grad=0.5
        self.beta = 0.01
        self.epslion = 0.2
        self.alpha_param = 0.5
        
        self.action_dim = len(self.dynamic_dict)
        self.asset_dim = self.states.shape[2]
        self.gamma = 0.999
        self.lr_actor = 3e-4        
        self.lr_critic = 1e-3  # Critic 학습률
        self.K_epochs = 80  # 정책 업데이트 횟수 몇번 샘플을 반복해서 업데이트 할지
        self.eps_clip = 0.2  # PPO 클리핑 범위
        self.has_continuous_action_space = True  # 연속적 행동 공간 여부
        self.action_std_init = 0.6  # 행동 표준 편차 초기값
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.num_agents = 2
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]
        self.save_path = f'./TARN_MAPPO/model/mulagent_{self.num_agents}_actor_lr{self.lr_actor}_critic_lr{self.lr_critic}_gamma{self.gamma}_epochs{self.epochs}_update{self.K_epochs}_eps_clip{self.eps_clip}_ppo_model_.pth'
        self.hyperparameters = hyperparameters
        self.patience = 10
        self.entropy_weight = 0.01
        self.mini_batch_size = 252
        self.all_weight_asset_dict = {dm : {tick : 0 for tick in self.real_data.tic.unique()} for dm in list(self.dynamic_dict.keys())}

        self.n_weight_dict = {tick : 0 for tick in self.real_data.tic.unique()}


class Stock_Env(EnvConfig):
    def __init__(self, config, eval = False):
        # 상속받은 EnvConfig 초기화
        super().__init__(config.states, config.real_data, config.real_numpy_data, config.hyperparameters, config.device)
        # self.current_idx = 0
        self.window_size = 20
        self.env_name = "Portfolio Allocation"
        # self.window_states = self.states[str(self.current_idx)]
        self.days = list(config.real_numpy_data.keys())
        # self.window_real_data = config.real_data.loc[self.days]
        self.max_step = 1000000
        self.update_interval = config.update_interval
        # self.update_interval = 10
        self.device = config.device

        self.save_path = config.save_path
        self.save_dict = {}
        self.save_n_weight_dict = {}
        # self.rewards = [self.invest_money]
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]
        self.num_agents = config.num_agents
        self.eval = eval

        self.idx = 0
        self.action_std_init = config.action_std_init
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.K_epochs = config.K_epochs
        self.eps_clip = config.eps_clip
        self.gamma = config.gamma
        self.reward_cond = config.reward_cond
        self.K_epochs = config.K_epochs
        self.entropy_weight = config.entropy_weight
        # self.epochs = config.epochs

        # 자산/보상 관리 (에이전트마다 독립)
        self.rewards = [config.invest_money]
        self.asset_weight_dict_list = {tick: 0.0 for tick in config.real_data.tic.unique()}
        self.n_rewards = [self.n_weight_money]
        self.all_weight_invest_dict = {dm : [10000000] for dm in self.dynamic_dict.keys()}
        self.all_weight_invest_rewards = {dm : [10000000] for dm in self.dynamic_dict.keys()}

        self.invest_money = 10000000
        if eval:
            self.days = list(self.real_data.index.unique())
            self.max_step = config.real_data.index.nunique()
        else:
            # self.max_step = config.max_step
            self.days = self.days[:self.max_step]
        
        
        
    def reset(self):
        """에이전트 수만큼 초기 상태를 반환"""
        if self.eval:
            self.idx = 0
        else:
            self.idx = random.randint(0, len(self.days) - self.update_interval- self.window_size)
        # self.idx = 51
        # self.idx = 1429
        # 투자금/weights 초기화
        self.rewards = [10000000]

        self.invest_money = 10000000
        self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.tic.unique()}
        # window_states 업데이트
        # self.gamma_reward = [0.0 for _ in range(self.num_agents)]
        
        self.current_step = 0  # 현재 `update_interval` 내에서 몇 번째 스텝인지 추적

        return self.states[self.idx]  # shape: [n_agents, feature_dim]


    # def reset(self):
    #     # 환경 초기화 함수
    #     self.idx = 0
    #     self.save_dict = {}
    #     self.rewards = []
    #     self.gamma_reward = 0.0
    #     self.invest_money = 10000000
    #     self.rewards.append(self.invest_money)
    #     return self.window_states[self.idx]
    
    def softmax(self, actions_weights, temperature):
        
        
        actions_weights = actions_weights / temperature  # Temperature Scaling 적용

        exp_x = np.exp(actions_weights - np.max(actions_weights))  # 안정적인 계산을 위한 조정
        return exp_x / exp_x.sum()
    
    def top3_action(self, actions, temperature, select_action=False):
        # actions에서 상위 3개의 인덱스를 반환하는 함수
        # top3_action = self.softmax(actions, temperature)
        # print("softmax_top_3", top3_action)
        if select_action:
            top3_indices = np.argsort(actions)[-3:]
            actions = np.zeros_like(actions)
            actions[top3_indices] = 1/3
        
        
        keys = list(self.dynamic_dict.keys())
        action_dict = {key: value for key, value in zip(keys, actions)}
        filtered_data = {key: value for key, value in action_dict.items() if value != 0}
        return filtered_data
    
    def calculate_action_return(self, top3_action, day):
        """
        top3 동적 자산배분 return을 구하는 코드입니다.
        """
        
        real_data_day = self.real_data.loc[day]
        future_days = self.days[self.idx+self.window_size]
        # print(self.days[self.idx+self.window_size])
        # future_days = list(self.days)[self.idx+1]
        future_data = self.real_data.loc[future_days]
        before_asset_dict = copy.deepcopy(self.asset_weight_dict_list)
        after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        for strategy, portfolio_weight in top3_action.items():
            # print(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight))
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
            # strategy_return = list(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight).values())
            # imp_dict[strategy] = strategy_return
        backtesting_investment, total_transaction_fee = dp.calculate_rebalancing_investment(real_data_day, future_data, before_asset_dict, after_asset_dict, self.invest_money)
        
        self.asset_weight_dict_list = after_asset_dict
        
        return backtesting_investment, total_transaction_fee
    
    
    def calculate_action_eval_return(self, top3_action, day):
        """
        top3 동적 자산배분 return을 구하는 코드입니다.
        """
        
        real_data_day = self.real_data.loc[day]
        future_data = self.real_data.loc[self.next_day]
        # print(self.days[self.idx+self.window_size])
        # future_days = list(self.days)[self.idx+1]
        # future_data = self.real_data.loc[future_days]
        before_asset_dict = copy.deepcopy(self.asset_weight_dict)
        after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        for strategy, portfolio_weight in top3_action.items():
            # print(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight))
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
            # strategy_return = list(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight).values())
            # imp_dict[strategy] = strategy_return
        backtesting_investment, total_transaction_fee = dp.calculate_rebalancing_investment(real_data_day, future_data, before_asset_dict, after_asset_dict, self.invest_money)
        
        self.asset_weight_dict = after_asset_dict
        
        return backtesting_investment, total_transaction_fee


    def calculate_n_weight_action_return(self, n_weight, day):
        """
        n_weight 동적 자산배분 return을 구하는 코드입니다.
        """
        
        real_data_day = self.real_data.loc[day]
        future_data = self.real_data.loc[self.next_day]

        before_asset_dict = copy.deepcopy(self.n_weight_dict)
        after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        for strategy, portfolio_weight in n_weight.items():
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
        backtesting_investment, total_transaction_fee = dp.calculate_rebalancing_investment(real_data_day, future_data, 
                                                                     before_asset_dict, after_asset_dict, self.n_weight_money)
        
        self.n_weight_dict = after_asset_dict
        
        return backtesting_investment, total_transaction_fee
    
    
    def calculate_all_weight_action_return(self, all_weight, day):
        """
        all weight 동적 자산배분 return을 구하는 코드입니다.
        """
        real_data_day = self.real_data.loc[day]
        # future_days = self.days[self.idx+self.window_size]
        future_data = self.real_data.loc[self.next_day]
        
        # 전략과 가중치 나옴
        for strategy, portfolio_weight in all_weight.items():
            before_asset_dict = copy.deepcopy(self.all_weight_asset_dict[strategy])
            after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
            backtesting_investment, total_transaction_fee = dp.calculate_rebalancing_investment(real_data_day, future_data, before_asset_dict, after_asset_dict, self.all_weight_invest_rewards[strategy][-1])
            self.all_weight_invest_dict[strategy] = total_transaction_fee

            self.all_weight_asset_dict[strategy] = after_asset_dict
            
        return self.all_weight_invest_dict



    def n_weight(self):
        n_weight_dict = {}
        keys = list(self.dynamic_dict.keys())    
        for key in keys:
            n_weight_dict[key] = 1/len(keys)
        return n_weight_dict
    
    
    def all_weight(self):
        all_weight_dict = {}
        keys = list(self.dynamic_dict.keys())    
        for key in keys:
            all_weight_dict[key] = 1.0
        return all_weight_dict
        
        
    def make_daily_data(self, transaction_fee):
        cash_weight = 1 - sum(self.asset_weight_dict_list.values())
        invest_weight = sum(self.asset_weight_dict_list.values())
        tot_money = self.invest_money - transaction_fee

        invest_money = tot_money * invest_weight

        cash_money = invest_money * cash_weight
        
        

        #daily_data_inx = self.real_data.loc[list(self.days)[self.idx]:self.days[self.idx+self.window_size]]
        daily_data_idx = list(self.real_data.loc[list(self.days)[self.idx]:self.days[self.idx + self.window_size-1]].index.unique())
        
        
        save_data = {date: 0 for date in daily_data_idx}
        for tic, weight in self.asset_weight_dict_list.items():
            if weight == 0:
                continue
#            imp = daily_data_inx[daily_data_inx["tic"] == tic]
            imp = self.real_data[self.real_data["tic"] == tic].loc[daily_data_idx, "close"]  # 해당 자산의 가격 데이터
            bench_mark = self.real_data[self.real_data["tic"] == tic].loc[daily_data_idx, "dji"]

            close_prices = imp.to_numpy()  # 종가를 NumPy 배열로 변환
            pct_changes = np.concatenate(([0], np.diff(close_prices) / close_prices[:-1]))  # NumPy 활용
            cumulative_returns = np.cumprod(1 + pct_changes)  # cumprod() 사용
            investment_values = invest_money * weight * cumulative_returns  # NumPy 벡터 연산

            # test = (1+imp["close"].pct_change().fillna(0)).cumprod() * invest_money * weight
        # 날짜별 투자 금액 업데이트 (for 루프 대신 zip 활용)
            for date, value in zip(daily_data_idx, investment_values):
                save_data[date] += value
        # 현금 투자 금액 추가
        for date in daily_data_idx:
            save_data[date] += cash_money
        return save_data, bench_mark
            
        
    
    
    def step(self, actions):
        # 행동을 받아서 다음 상태, 보상, 에피소드 종료 여부를 반환하는 함수
        # idx에 해당하는 날짜를 받아옴
        day = self.days[self.idx]
        print(day)
        # print("actions", actions)
        # state를 가져옴
        try:
            print(self.idx)
                # print(self.days[self.idx+self.window_size])
        except:
            print("End of data")
        state = self.states[self.idx]
        reward_dict = {}


        action_1d = actions
        top3_action = self.top3_action(action_1d, 2.0)
        rebalance_investment, total_transaction_fee = self.calculate_action_return(top3_action, day)
        self.rewards.append(rebalance_investment)
        self.invest_money = rebalance_investment
        imp, bench_mark = self.make_daily_data(total_transaction_fee)
            # invest_list = imp[self.days[self.idx + self.window_size-1]]
            # imp = pd.DataFrame(list(imp.items()), columns=["date", "invest"])
            # imp["date"] = pd.to_datetime(imp["date"])  # 날짜 타입 변환
            # imp.set_index("date", inplace=True)  # 날짜를 인덱스로 설정
            # self.invest_money_list[i] = invest_list
        backtest_invest_money = list(imp.values())
        returns = backtest.calculate_return(backtest_invest_money)
        bench_return = backtest.calculate_return(bench_mark)
        sharpe = backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 252)
        frequency = (returns > bench_return).sum() / len(returns)       
        sortino = backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 252)
        calmar = backtest.calculate_calmar_ratio(returns, annual_factor=252)
        mdd = backtest.calculate_max_drawdown(returns)
            
        combined = sharpe + sortino + calmar
        combined_contest = frequency + sharpe - mdd
        reward_dict = {'return': returns, 'sharpe' : sharpe, 'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 
                            'combined' : combined, "combined_contest": combined_contest, "frequency": frequency}




        # asset_reward = sum(imp_dict.values())
            # self.rewards.append(self.invest_money)
        # self.save_dict[day] = rebalance_investment
        
        # done = (self.idx+1 >= self.max_step)
            
            # if len(self.rewards[i]) < 2: # reward가 1개 이하인 경우 sharpe를 구할 수 없기에
            #     returns = backtest.calculate_return(self.rewards[i])

            #     reward_dict[i] = {'return': returns, 'sharpe': 0.0, 'sortino': 0.0, "calmar":0.0, 'mdd': 0.0, 'combined' : 0.0}
        
            # else:
            #     returns = backtest.calculate_return(self.rewards[i])
            #     sharpe = backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
            #     sortino = backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
            #     calmar = backtest.calculate_calmar_ratio(returns, annual_factor=12)
            #     mdd = backtest.calculate_max_drawdown(returns)
            #     combined = sharpe + sortino + calmar
            #     reward_dict[i] = {'return': returns, 'sharpe' : sharpe, 'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 'combined' : combined}
        
        self.idx += 1


        self.current_step += 1
        # done = self.idx >= self.update_interval
        # 127
        done = self.current_step >= self.update_interval


        if done:
            print("Episode Done")
            reward_li = reward_dict[self.reward_cond]
            self.reset()
            return state, reward_li, done, self.invest_money_list, reward_dict

        else:
            day = list(self.days)[self.idx]
            state = self.states[self.idx]
            self.invest_money_list = 10000000
            self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.tic.unique()}

            return state, reward_dict[self.reward_cond], done, self.invest_money_list, reward_dict
    

    def make_daily_invest_data(self, empty_df, asset_weight, invest_money, daily_data, total_transaction_fee,  day, next_day):
            cash_weight = 1- sum(asset_weight.values())
            invest_weight = sum(asset_weight.values())
            invest_money = invest_money - total_transaction_fee
            cash_money = invest_money * cash_weight
            # invest_money_remain = invest_money * invest_weight
            
            daily_data_inx = self.real_data.loc[day:next_day]
            # next_day = daily_data_inx.index.unique()[-2]
            # daily_data_inx = self.real_data.loc[day:next_day]
    
            
            save_data = pd.DataFrame(index = daily_data_inx.index.unique(), columns=["invest"]).fillna(0)
            # print("after invest_money : ", invest_money_remain)
            # print("cash_money : ", cash_money)
            # print("total_invest_money : ", cash_money + invest_money_remain)
            # print(daily_data.TICKER.nunique())
            # print(len(asset_weight.keys()))
            for tic, weight in asset_weight.items():
                if weight == 0:
                    continue
                imp = daily_data[daily_data["tic"] == tic]
                daily_return = imp["close"].pct_change().fillna(0)
                daily_return = daily_return.loc[day:next_day]
                test = weight * (1+daily_return).cumprod() * invest_money
                save_data["invest"] += test.values
            save_data["invest"] += cash_money
            # print(save_data)
            empty_df = pd.concat([empty_df, save_data], axis=0)
            return empty_df
        

    
    def eval_step(self, weights, empty_df, all_weight_df, n_weight_df):
        """test를 평가하는 함수입니다.

        Args:
            weights (_type_): _description_
        """
        
        day = list(self.days)[self.idx]
        # print(day)
        state = self.states[self.idx]

        if self.idx+self.window_size <= self.max_step:
            self.next_day = self.days[self.idx+self.window_size-1]
        else:
            self.next_day = self.real_data.index.unique()[-1]  # 전체 데이터의 마지막 날짜
        print(f"Day: {day} ~ {self.next_day}")      

        
        
        n_weight_dict = self.n_weight()
        all_weight_dict = self.all_weight()
        print(weights)
        top3_action = self.top3_action(weights, 1.0, select_action = False)
        print(top3_action)
        
        daily_money = self.invest_money
        
        rebalance_investment_top3, total_transaction_fee = self.calculate_action_eval_return(top3_action, day)
        empty_df = self.make_daily_invest_data(empty_df, self.asset_weight_dict, daily_money, self.real_data, total_transaction_fee, day, self.next_day)

        self.invest_money = empty_df.iloc[-1].values[0]
        
        
        daily_nweight_money = self.n_weight_money
        rebalance_investment_n_weight, n_total_transaction_fee = self.calculate_n_weight_action_return(n_weight_dict, day)
        n_weight_df = self.make_daily_invest_data(n_weight_df,  self.n_weight_dict, daily_nweight_money, self.real_data, n_total_transaction_fee, day, self.next_day)
        # self.n_weight_money = rebalance_investment_n_weight
        self.n_weight_money = n_weight_df.iloc[-1].values[0]
        # 전략별 투자
        all_invest_dict = self.all_weight_invest_dict.copy()
        rebalance_invest_all_weight = self.calculate_all_weight_action_return(all_weight_dict, day)

        dm_df = pd.DataFrame()
        for dm, all_weight_asset in all_weight_dict.items():
            imp_df = pd.DataFrame()

            all_dm_money = all_invest_dict[dm] # 비중
            strategy_asset_dict =  self.all_weight_asset_dict[dm] # 전략별 자산 비중
            dm_fee = rebalance_invest_all_weight[dm]
            # print(dm, strategy_asset_dict)
            # print("all_dm_money", all_dm_money)
            imp_df = self.make_daily_invest_data(imp_df, strategy_asset_dict, all_dm_money, self.real_data, dm_fee, day, self.next_day)
            imp_df.columns = [f"{dm}"]
            # print(dm_df)
            dm_df = pd.concat([dm_df, imp_df], axis=1)
        # print("dm_df")
        # print(dm_df)
        all_weight_df = pd.concat([all_weight_df, dm_df])
        # print(self.all_weight_dict)
        self.all_weight_invest_dict = all_weight_df.iloc[-1].to_dict()
        self.rewards.append(self.invest_money)
        self.n_rewards.append(self.n_weight_money)
        
        
        
        
        for dm, all_weight_dict in self.all_weight_invest_dict.items():
            self.all_weight_invest_rewards[dm].append(all_weight_dict)
                

        if len(self.rewards) < 2: # reward가 1개 이하인 경우 sharpe를 구할 수 없기에
            returns = backtest.calculate_return(self.rewards)
            reward_dict = {'return': returns, 'sharpe': 0.0, 'sortino': 0.0, "calmar":0.0, 'mdd': 0.0, 'combined' : 0.0}
    
        else:
            returns = backtest.calculate_return(self.rewards)
            sharpe = backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
            sortino = backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
            calmar = backtest.calculate_calmar_ratio(returns, annual_factor=12)
            mdd = backtest.calculate_max_drawdown(returns)
            combined = sharpe + sortino + calmar
            reward_dict = {'return': returns, 'sharpe' : sharpe, 'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 'combined' : combined}

        
               
        self.idx += self.window_size
        done = self.idx >= self.max_step

        # self.gamma_reward = (reward_dict[self.reward_cond])  + self.gamma * self.gamma_reward
        if done:
            
            print("Episode Done")
            reward_li = self.rewards
            asset_weight_dict_li = self.asset_weight_dict
                
            self.reset()
            return state, reward_li, done, self.n_rewards,  self.all_weight_invest_dict, asset_weight_dict_li, top3_action, empty_df, all_weight_df, n_weight_df


        else:
            # self.idx += 1
            day = list(self.days)[self.idx]
            # self.days[self.idx+self.window_size-1]
            state = self.states[self.idx]

            return state, self.rewards, done, self.n_rewards,  self.all_weight_invest_rewards, self.asset_weight_dict, top3_action, empty_df, all_weight_df, n_weight_df
    
    
    
    
if __name__ == "__main__":
    import torch
    import copy
    import pandas as pd
    train_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    train_dataset = pd.read_csv("/Users/pjy97/Desktop/hyu/research/RL/code/data/train_data.csv", index_col=0)
    test_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    env = copy.deepcopy(Stock_Env(train_tensor, train_dataset))
