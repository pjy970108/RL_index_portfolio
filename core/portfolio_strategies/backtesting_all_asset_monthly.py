# backtest_module.py

import pandas as pd

from dynamic_portfolio_monthly import *

def run_partial_backtests(
    df_long: pd.DataFrame,
    current_date: pd.Timestamp,
    top_n: int = 10,
    look_backs: int = 252,
    rebalance_every: int = 20,
    cost: float = 0.003,
    top_pct: float = 0.1,
    risk_coefficient: float = 2,
    risk_free_rate : float = 0.0
    
) -> dict:
    """
    현재 시점까지만 각 전략별 백테스트를 수행하여 전략별 수익률 시리즈를 반환

    Parameters:
    - df_long: long-form 포맷의 데이터프레임
    - current_date: 현재 날짜 기준

    Returns:
    - strategy_returns: 전략 이름 -> 수익률 시리즈 딕셔너리
    """
    df = df_long.sort_values(['date', 'ticker']) if 'date' in df_long.columns else df_long.sort_index()
    df['date'] = df.index
    df = df.set_index('date')
    date_list = df.index.drop_duplicates().sort_values()

    if current_date not in date_list:
        return {}

    current_idx = date_list.get_loc(current_date)
    start_idx = max(0, current_idx - look_backs)
    end_idx = current_idx + rebalance_every
    date_range = date_list[start_idx:end_idx]
    df = df[df.index.isin(date_range)]
    
    # if len(df.index.unique()) < look_backs:
    #     return {}  # 데이터 부족 시 빈 결과 반환
    df = df.reset_index()

    pivot_close = df.pivot(index='date', columns='ticker', values='close')
    pivot_return = pivot_close.pct_change().fillna(0)
    pivot_score = df.pivot(index='date', columns='ticker', values='mom_score')
    pivot_paa_score = df.pivot(index='date', columns='ticker', values='paa_score')
    pivot_sma = df.pivot(index='date', columns='ticker', values='SMA_220')

    strategies = {}

    try:
        strategies['RP'] = backtest_strategy(pivot_return, compute_risk_parity_weight_from_window, rebalance_every, look_backs, cost)
        strategies['Min_Var'] = backtest_strategy(pivot_return, compute_minvar_weights, rebalance_every, look_backs, cost)
        strategies['Mean_Var_max_sharpe'] = backtest_strategy(pivot_return, compute_max_sharpe_min_var, rebalance_every, look_backs, cost, risk_free_rate=risk_free_rate)
        strategies['DAA'] = backtest_daa_from_pivot(pivot_return, pivot_score, look_backs, rebalance_every, top_n, 0.0, cost)
        strategies['PAA'] = backtest_paa_from_pivot(pivot_return, pivot_paa_score, look_backs, rebalance_every, top_n, 0.0, cost)
        strategies['GTAA'] = backtest_gtaa_from_pivot(pivot_return, pivot_close, pivot_sma, look_backs, rebalance_every, top_n, cost)
        strategies['equal'] = backtest_equal_weight_20day(pivot_return, rebalance_every, look_backs, cost)
    except Exception as e:
        print(f"[Backtest Error @ {current_date}] {e}")

    # 수익률 시리즈만 반환 (weight는 제외)
    strategy_returns = {name: ret[0] for name, ret in strategies.items()}
    return strategy_returns



def run_partial_backtests_v2(
    index_long: pd.DataFrame,
    future_long: pd.DataFrame,
    equal_long : pd.DataFrame,
    current_date: pd.Timestamp,
    top_n: int = 10,
    look_backs: int = 252,
    rebalance_every: int = 20,
    cost: float = 0.003,
    top_pct: float = 0.1,
    risk_coefficient: float = 2,
    risk_free_rate : float = 0.0
    
) -> dict:
    """
    현재 시점까지만 각 전략별 백테스트를 수행하여 전략별 수익률 시리즈를 반환

    Parameters:
    - df_long: long-form 포맷의 데이터프레임
    - current_date: 현재 날짜 기준

    Returns:
    - strategy_returns: 전략 이름 -> 수익률 시리즈 딕셔너리
    """
    index_df = index_long.sort_values(['date', 'ticker']) if 'date' in index_long.columns else index_long.sort_index()
    future_df = future_long.sort_values(['date', 'ticker']) if 'date' in future_long.columns else future_long.sort_index()
    equal_df = equal_long.sort_values(['date', 'ticker']) if 'date' in equal_long.columns else equal_long.sort_index()
    index_df['date'] = index_df.index
    future_df['date'] = future_df.index
    # equal_df['date'] = equal_df.index
    index_df = index_df.set_index('date')
    future_df = future_df.set_index('date')
    # equal_df = equal_df.reset_index('date')
    
    date_list = index_df.index.drop_duplicates().sort_values()

    if current_date not in date_list:
        return {}

    current_idx = date_list.get_loc(current_date)
    start_idx = max(0, current_idx - look_backs)
    end_idx = current_idx + rebalance_every
    date_range = date_list[start_idx:end_idx]
    index_df = index_df[index_df.index.isin(date_range)]
    future_df = future_df[future_df.index.isin(date_range)]
    equal_df = equal_df[equal_df.index.isin(date_range)]
    
    # if len(df.index.unique()) < look_backs:
    #     return {}  # 데이터 부족 시 빈 결과 반환
    # index_df = index_df.reset_index()
    # future_df = future_df.reset_index()
    equal_df = equal_df.reset_index()
    
    # index_pivot_close = index_df.pivot(index='date', columns='ticker', values='close')
    # index_pivot_return = index_pivot_close.pct_change().fillna(0)
    # future_pivot_close = future_df.pivot(index='date', columns='ticker', values='close')
    # future_pivot_return = future_pivot_close.pct_change().fillna(0)
    equal_pivot_close = equal_df.pivot(index='date', columns='ticker', values='close')
    equal_pivot_return = equal_pivot_close.pct_change().fillna(0)
    
    # index_pivot_paa_score = index_df.pivot(index='date', columns='ticker', values='paa_score')
    # future_pivot_paa_score = future_df.pivot(index='date', columns='ticker', values='paa_score')
    all_pivot_paa_score = equal_df.pivot(index='date', columns='ticker', values='paa_score')

    strategies = {}

    try:
        strategies['risk_parity'] = backtest_strategy(equal_pivot_return, compute_risk_parity_weight_from_window, rebalance_every, look_backs, cost)
        
        strategies['min_var'] = backtest_strategy(equal_pivot_return, compute_minvar_weights, rebalance_every, look_backs, cost)
        strategies['max_sharpe'] = backtest_strategy(equal_pivot_return, compute_max_sharpe_min_var, rebalance_every, look_backs, cost, risk_free_rate=risk_free_rate)
        strategies['paa'] = backtest_paa_from_pivot(equal_pivot_return, all_pivot_paa_score, look_backs, rebalance_every, top_n, 0.0, cost)

        strategies['equal'] = backtest_equal_weight_20day(equal_pivot_return, rebalance_every, look_backs, cost)
    except Exception as e:
        print(f"[Backtest Error @ {current_date}] {e}")

    # 수익률 시리즈만 반환 (weight는 제외)
    strategy_returns = {name: ret[0] for name, ret in strategies.items()}
    return strategy_returns








def eval_partial_backtests_v2(
    index_long: pd.DataFrame,
    future_long: pd.DataFrame,
    equal_long : pd.DataFrame,
    trading_data: tuple,
    top_n: int = 10,
    look_backs: int = 252,
    rebalance_every: int = 20,
    cost: float = 0.003,
    top_pct: float = 0.1,
    risk_coefficient: float = 2,
    risk_free_rate : float = 0.0
    
) -> dict:
    """
    현재 시점까지만 각 전략별 백테스트를 수행하여 전략별 수익률 시리즈를 반환

    Parameters:
    - df_long: long-form 포맷의 데이터프레임
    - current_date: 현재 날짜 기준

    Returns:
    - strategy_returns: 전략 이름 -> 수익률 시리즈 딕셔너리
    """
    index_df = index_long.sort_values(['date', 'ticker']) if 'date' in index_long.columns else index_long.sort_index()
    future_df = future_long.sort_values(['date', 'ticker']) if 'date' in future_long.columns else future_long.sort_index()
    equal_df = equal_long.sort_values(['date', 'ticker']) if 'date' in equal_long.columns else equal_long.sort_index()

    index_df['date'] = index_df.index
    future_df['date'] = future_df.index
    
    index_df = index_df.set_index('date')
    future_df = future_df.set_index('date')

    date_list = index_df.index.drop_duplicates().sort_values()

    # if trading_data not in date_list:
    #     return {}
    start_date = trading_data[0]
    end_date = trading_data[1]
    current_idx = date_list.get_loc(start_date)
    start_idx = max(0, current_idx - look_backs)
    end_idx = date_list.get_loc(end_date)
    date_range = date_list[start_idx:end_idx+1]
    # index_df = index_df[index_df.index.isin(date_range)]
    # future_df = future_df[future_df.index.isin(date_range)]
    equal_df = equal_df[equal_df.index.isin(date_range)]
    
    # if len(df.index.unique()) < look_backs:
    #     return {}  # 데이터 부족 시 빈 결과 반환
    # index_df = index_df.reset_index()
    # future_df = future_df.reset_index()
    equal_df = equal_df.reset_index()
    
    # index_pivot_close = index_df.pivot(index='date', columns='ticker', values='close')
    # index_pivot_return = index_pivot_close.pct_change().fillna(0)
    # future_pivot_close = future_df.pivot(index='date', columns='ticker', values='close')
    # future_pivot_return = future_pivot_close.pct_change().fillna(0)
    equal_pivot_close = equal_df.pivot(index='date', columns='ticker', values='close')
    equal_pivot_return = equal_pivot_close.pct_change().fillna(0)
    all_pivot_paa_score = equal_df.pivot(index='date', columns='ticker', values='paa_score')

    # index_pivot_paa_score = index_df.pivot(index='date', columns='ticker', values='paa_score')
    # future_pivot_paa_score = future_df.pivot(index='date', columns='ticker', values='paa_score')
    
    strategies = {}

    try:
                
        strategies['risk_parity'] = backtest_strategy(equal_pivot_return, compute_risk_parity_weight_from_window, rebalance_every, look_backs, cost, rebalance_every)
        
        strategies['min_var'] = backtest_strategy(equal_pivot_return, compute_minvar_weights, rebalance_every, look_backs, cost, rebalance_every)
        strategies['max_sharpe'] = backtest_strategy(equal_pivot_return, compute_max_sharpe_min_var, rebalance_every, look_backs, cost, rebalance_every, risk_free_rate=risk_free_rate)
        strategies['paa'] = backtest_paa_from_pivot(equal_pivot_return, all_pivot_paa_score, look_backs, rebalance_every, top_n, 0.0, cost,rebalance_every)

        strategies['equal'] = backtest_equal_weight_20day(equal_pivot_return, rebalance_every, look_backs, cost, rebalance_every)
        print(len(strategies['equal'][0]))
 
        
    except Exception as e:
        print(f"[Backtest Error @ {trading_data}] {e}")

    # 수익률 시리즈만 반환 (weight는 제외)
    strategy_returns = {name: ret[0] for name, ret in strategies.items()}
    strategy_weight = {name: ret[1] for name, ret in strategies.items()}

    return strategy_returns, strategy_weight



def eval_partial_backtests(
    df_long: pd.DataFrame,
    trading_data: tuple,
    top_n: int = 10,
    look_backs: int = 252,
    rebalance_every: int = 20,
    cost: float = 0.003,
    top_pct: float = 0.1,
    risk_coefficient: float = 2,
    risk_free_rate : float = 0.0
    
) -> dict:
    """
    현재 시점까지만 각 전략별 백테스트를 수행하여 전략별 수익률 시리즈를 반환

    Parameters:
    - df_long: long-form 포맷의 데이터프레임
    - current_date: 현재 날짜 기준

    Returns:
    - strategy_returns: 전략 이름 -> 수익률 시리즈 딕셔너리
    """
    df = df_long.sort_values(['date', 'ticker']) if 'date' in df_long.columns else df_long.sort_index()
    df['date'] = df.index
    df = df.set_index('date')
    date_list = df.index.drop_duplicates().sort_values()

    # if trading_data not in date_list:
    #     return {}
    start_date = trading_data[0]
    end_date = trading_data[1]
    current_idx = date_list.get_loc(start_date)
    start_idx = max(0, current_idx - look_backs)
    end_idx = date_list.get_loc(end_date)
    date_range = date_list[start_idx:end_idx+1]
    df = df[df.index.isin(date_range)]
    
    # if len(df.index.unique()) < look_backs:
    #     return {}  # 데이터 부족 시 빈 결과 반환
    df = df.reset_index()

    pivot_close = df.pivot(index='date', columns='ticker', values='close')
    pivot_return = pivot_close.pct_change().fillna(0)
    pivot_score = df.pivot(index='date', columns='ticker', values='mom_score')
    pivot_paa_score = df.pivot(index='date', columns='ticker', values='paa_score')
    pivot_sma = df.pivot(index='date', columns='ticker', values='SMA_220')

    strategies = {}

    try:
        strategies['RP'] = backtest_strategy(pivot_return, compute_risk_parity_weight_from_window, rebalance_every, look_backs, cost)
        strategies['Min_Var'] = backtest_strategy(pivot_return, compute_minvar_weights, rebalance_every, look_backs, cost)
        strategies['Mean_Var_max_sharpe'] = backtest_strategy(pivot_return, compute_max_sharpe_min_var, rebalance_every, look_backs, cost, risk_free_rate=risk_free_rate)
        strategies['DAA'] = backtest_daa_from_pivot(pivot_return, pivot_score, look_backs, rebalance_every, top_n, 0.0, cost)
        strategies['PAA'] = backtest_paa_from_pivot(pivot_return, pivot_paa_score, look_backs, rebalance_every, top_n, 0.0, cost)
        strategies['GTAA'] = backtest_gtaa_from_pivot(pivot_return, pivot_close, pivot_sma, look_backs, rebalance_every, top_n, cost)
        strategies['equal'] = backtest_equal_weight_20day(pivot_return, rebalance_every, look_backs, cost)
    except Exception as e:
        print(f"[Backtest Error @ {trading_data}] {e}")

    # 수익률 시리즈만 반환 (weight는 제외)
    strategy_returns = {name: ret[0] for name, ret in strategies.items()}
    strategy_weight = {name: ret[1] for name, ret in strategies.items()}

    return strategy_returns, strategy_weight
