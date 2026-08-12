import os
import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)
import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "base_line_model/Task_1_FinRL_DeepSeek_Stock")))
# print("Current working directory:", os.getcwd())
# print(sys.path)
import dynamic_portfolio as dp
from backtesting import run_partial_backtests  # 함수만 명시적으로 import
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import torch
import copy
import random
import pandas as pd
import eval_metric



class Stock_Env:
    def __init__(self, config, states, real_df, hyperparameters, eval=False):
        # 상속받은 EnvConfig 초기화
        # self.current_idx = 0
        
        self.states = states
        self.real_data = real_df
        self.hyperparameters = hyperparameters
        self.eval = eval
        
        self.env_name = config.get("env_name", "Portfolio Allocation")
        self.window_size = config.get("window_size", 20)
        self.reward_cond = config.get("reward_cond", "combined")
        self.top_n = config.get("top_n", 3)
        self.look_back = config.get("look_back", 252)
        self.cost = config.get("cost", 0.003)
        self.top_pct = config.get("top_pct", 0.1)

        self.dynamic_dict = {
            "RP": dp.backtest_strategy,
            "Min_Var": dp.backtest_strategy,
            "Mean_Var_max_sharpe": dp.backtest_strategy,
            "DAA": dp.backtest_daa_from_pivot,
            "PAA": dp.backtest_paa_from_pivot,
            "GTAA": dp.backtest_gtaa_from_pivot
        }
        self.update_interval = config["update_interval"]
        self.num_trajectories = config["num_trajectories"]
        self.device = config["device"]
        self.save_path = config["save_path"]
        # self.num_trajectories_map = config["num_trajectories_map"]
        self.num_workers = config.get("num_workers", 4)

        self.num_iterations = config["num_iterations"]
        self.num_steps = config["num_steps"]
        self.batch_samples = config["batch_samples"]
        self.num_workers = config.get("num_workers", 4)
        self.mu = config["mu"]
        self.clip_grad = config["clip_grad"]
        self.beta = config["beta"]
        self.epsilon = config["epsilon"]
        self.alpha_param = config["alpha_param"]
        self.action_dim = len(self.dynamic_dict)
        self.asset_dim = self.states.shape[2]
        self.gamma = config["gamma"]
        self.lr_actor = config["lr_actor"]
        self.eps_clip = config["eps_clip"]
        self.has_continuous_action_space = config["has_continuous_action_space"]
        self.action_std_init = config["action_std_init"]
        self.entropy_weight = config["entropy_weight"]
        self.patience = config["patience"]
        self.mini_batch_size = config["mini_batch_size"]
        

        self.asset_weight_dict = {tick: 0 for tick in self.real_data.ticker.unique()}
        self.all_weight_asset_dict = {dm: {tick: 0 for tick in self.real_data.ticker.unique()} for dm in self.dynamic_dict.keys()}
        self.n_weight_dict = {tick: 0 for tick in self.real_data.ticker.unique()}
        
        if eval:
            self.days = list(self.real_data.index.unique())
            self.max_step = config.real_data.index.nunique()
        else:
            self.max_step = config["update_interval"]
            self.days = list(self.real_data.index.unique()[self.look_back:])
                
        self.idx = 0
        self.current_step = 0
        self.save_dict = {}
        self.save_n_weight_dict = {}


        self.rewards = [1.0]  # 초기 포트폴리오 가치 (수익률 기준)
        self.n_rewards = [1.0]
        self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.ticker.unique()}
        self.all_weight_invest_dict = {dm: [1.0] for dm in self.dynamic_dict.keys()}
        self.all_weight_invest_rewards = {dm: [1.0] for dm in self.dynamic_dict.keys()}
        
        self.risk_free_rate = config.get("risk_free_rate", 0.0)
        self.annual_factor = config.get("annual_factor", 252)
        self.risk_coefficient = config.get("risk_coefficient", 10)
        self.top_pct = config.get("top_pct", 0.5)
        self.top_k = config.get("top_k", 5)
        self.temperatures = config.get("temperatures", 1.0)
        
        
    def reset(self):
        """에이전트 수만큼 초기 상태를 반환"""
        if self.eval:
            self.idx = 0
        else:
            self.idx = random.randint(0, len(self.days) - self.update_interval- self.window_size)
        # self.idx = 51
        # self.idx = 1429
        # 투자금/weights 초기화
        self.rewards = [1.0]  # 초기 포트폴리오 가치 (수익률 기준)
        self.n_rewards = [1.0]
        self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.ticker.unique()}
        self.current_step = 0


        return self.states[self.idx]  # shape: [n_agents, feature_dim]


    def group_reset(self):
        """에이전트 수만큼 초기 상태를 반환"""
        self.idx = self.idx - self.update_interval
        # self.idx = 51
        # self.idx = 1429
        # 투자금/weights 초기화
        self.rewards = [1.0]  # 초기 포트폴리오 가치 (수익률 기준)
        self.n_rewards = [1.0]
        self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.ticker.unique()}
        self.current_step = 0


        return self.states[self.idx]  # shape: [n_agents, feature_dim]

    
    def softmax(self, actions_weights, temperature):
        actions_weights = actions_weights / temperature  # Temperature Scaling 적용

        exp_x = np.exp(actions_weights - np.max(actions_weights))  # 안정적인 계산을 위한 조정
        return exp_x / exp_x.sum()
    
    def top3_action(self, actions):
        # actions에서 상위 3개의 인덱스를 반환하는 함수
        weight_actions = self.softmax(actions, self.temperatures)
        # print("softmax_top_3", top3_action)
        # if select_action:
        #     top3_indices = np.argsort(actions)[-3:]
        #     actions = np.zeros_like(actions)
        #     actions[top3_indices] = 1/3
        
        
        keys = list(self.dynamic_dict.keys())
        action_dict = {key: value for key, value in zip(keys, weight_actions)}
        filtered_data = {key: value for key, value in action_dict.items() if value != 0}
        return filtered_data
    
    
    def simulate_combined_portfolio_returns(self, top3_action: dict ) -> pd.Series:
        """
        전략별 일간 수익률과 가중치를 바탕으로 포트폴리오 수익률 계산

        Parameters:
        - strategy_returns_dict: 각 전략별 일간 수익률 Series 딕셔너리 {strategy_name: pd.Series}
        - top3_action: 선택된 전략과 해당 가중치 딕셔너리 {strategy_name: weight}
        - date_range: slice 객체 또는 (start_date, end_date) 튜플로 된 인덱스 범위

        Returns:
        - combined_returns: 전략 조합으로 만든 일간 수익률 Series
        """
        combined_returns = None
        equal_returns = None

        for strategy, strat_returns in self.strategy_returns.items():
            if strategy == "equal":
                equal_returns = strat_returns
                continue
            # if isinstance(date_range, slice):
            #     strat_returns = strat_returns.loc[date_range]
            # else:
            #     start_date, end_date = date_range
            #     strat_returns = strat_returns.loc[start_date:end_date]

            weighted_returns = strat_returns * top3_action[strategy]

            if combined_returns is None:
                combined_returns = weighted_returns.copy()
            else:
                combined_returns = combined_returns.add(weighted_returns, fill_value=0)

        return combined_returns, equal_returns

    
    
    def calculate_action_return(self, top3_action):
        
        combined_returns, equal_returns = self.simulate_combined_portfolio_returns(
            top3_action=top3_action
        )

        cumulative_ret = (1 + combined_returns).prod()
        invest_money = self.rewards[-1]
        invest_result = invest_money * cumulative_ret
        
        cumulative_equal = (1 + equal_returns).prod()
        equal_money = self.n_rewards[-1]
        equal_result = equal_money * cumulative_equal
        
        
        self.rewards.append(invest_result)
        self.n_rewards.append(equal_result)
        # 거래 비용 계산
        # transaction_cost = sum(
        #     np.abs(np.array(list(top3_action.values())) -
        #         np.array(list(self.prev_action.values())))
        # ) * self.cost

        # self.prev_action = top3_action
        # return invest_result, transaction_cost
        return combined_returns, equal_returns # daily_return, 누적 reward



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
    
    
    def step(self, actions, group_reset = False):
        # 행동을 받아서 다음 상태, 보상, 에피소드 종료 여부를 반환하는 함수
        # idx에 해당하는 날짜를 받아옴
        day = self.days[self.idx]
        print(day)
        # state를 가져옴
        try:
              print(self.idx)
        except:
            print("End of data")
        state = self.states[self.idx]
        reward_dict = {}
        # 2. 현재 날짜까지 전략 수익률 계산 (동적 백테스트)
        self.strategy_returns = run_partial_backtests(
            df_long=self.real_data,
            current_date=day,
            top_n=self.top_n,
            look_backs=self.look_back,
            rebalance_every=self.window_size,
            cost=self.cost,
            top_pct = self.top_pct,
            risk_coefficient=self.risk_coefficient
        )

        action_1d = actions[-1]
        top3_action = self.top3_action(action_1d)
        combined_returns, equal_returns = self.calculate_action_return(top3_action)

        # imp, bench_mark = self.make_daily_data(total_transaction_fee)
            # invest_list = imp[self.days[self.idx + self.window_size-1]]
            # imp = pd.DataFrame(list(imp.items()), columns=["date", "invest"])
            # imp["date"] = pd.to_datetime(imp["date"])  # 날짜 타입 변환
            # imp.set_index("date", inplace=True)  # 날짜를 인덱스로 설정
            # self.invest_money_list[i] = invest_list
        # backtest_invest_money = list(imp.values())
        model_eval_metric = eval_metric.calculate_performance_metrics(combined_returns, annual_factor = self.annual_factor, risk_free_rate= self.risk_free_rate)
        equal_eval_metric = eval_metric.calculate_performance_metrics(equal_returns, annual_factor = self.annual_factor, risk_free_rate= self.risk_free_rate)
        returns = eval_metric.calculate_annual_return(combined_returns, annual_factor = self.annual_factor)
        model_sharpe = model_eval_metric["sharpe"]
        model_sortino = model_eval_metric["sortino"]
        model_calmar = model_eval_metric["calmar"]
        combined = model_sharpe + model_sortino + model_calmar
        model_eval_metric["combined_reward"] = combined
        # reward_dict = {'return': returns, 'sharpe' : sharpe, 'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 
        #                     'combined' : combined}

        self.idx += 1


        self.current_step += 1
        # done = self.idx >= self.update_interval
        # 127
        done = self.current_step >= self.update_interval


        if done:
            print("Episode Done")
            reward_li = model_eval_metric[self.reward_cond]
            if group_reset:
                self.group_reset()
            else:
                self.reset()
            return state, reward_li, done, model_eval_metric

        else:
            day = list(self.days)[self.idx]
            state = self.states[self.idx]
            self.rewards = [1.0]  # 초기 포트폴리오 가치 (수익률 기준)
            self.n_rewards = [1.0]            
            self.asset_weight_dict_list = {tick: 0.0 for tick in self.real_data.ticker.unique()}

            return state, model_eval_metric[self.reward_cond], done, model_eval_metric
    

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
