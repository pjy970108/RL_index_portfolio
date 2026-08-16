import numpy as np
import pandas as pd
import cvxpy as cp


def compute_risk_parity_weight_from_window(returns_window):
    var = returns_window.var()
    inv_var = 1 / var
    weights = inv_var / inv_var.sum()
    return weights


def compute_minvar_weights(mu, cov_matrix):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value


def convert_to_periodic_returns(daily_returns: pd.Series, weights_record, period: int = 20) -> pd.Series:
    """
    일간 수익률을 주기별 누적 수익률로 변환

    Returns:
    - 기간 단위 누적 수익률 Series (index는 각 기간 마지막 날짜)
    """
    group_ids = np.arange(len(daily_returns)) // period
    periodic_returns = daily_returns.groupby(group_ids).apply(lambda x: (1 + x).prod() - 1)
    periodic_index = daily_returns.groupby(group_ids).apply(lambda x: x.index[-1])
    periodic_returns.index = periodic_index
    weights_record = weights_record.loc[periodic_index]
    return periodic_returns, weights_record


def compute_max_sharpe_min_var(mu, cov_matrix, risk_free_rate=0.0):
    n = len(mu)
    x = cp.Variable(n)
    excess_mu = mu - risk_free_rate
    w_tilde = cp.Variable(n)  # 치환된 weight (w_tilde = k * w)
    k = cp.Variable(nonneg=True)  # 스케일 변수
    objective = cp.Minimize(cp.quad_form(w_tilde, cov_matrix))

    constraints = [
        excess_mu.T @ w_tilde == 1,   # 초과수익률 고정
        cp.sum(w_tilde) == k,          # w_tilde = k * w
        w_tilde >= 0          # w_tilde = k * w
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if w_tilde.value is not None and k.value is not None and k.value > 0:
            w = w_tilde.value / k.value
            return w
        else:
            print("최적화 실패. 균등 포트폴리오로 fallback.")
            return np.ones(n) / n
    except cp.error.SolverError as e:
        print(f"최적화 오류: {e}")
        print("최적화 실패. 균등 포트폴리오로 fallback.")
        return np.ones(n) / n

def backtest_strategy(returns, compute_weights_fn, rebalance_every=20, window_days=252, cost=0.003, period = 20,**kwargs):
    dates = returns.index
    portfolio_returns = []
    weights_record = []
    prev_weights = None

    i = window_days
    while i < len(dates) - rebalance_every+1:
        window_data = returns.iloc[i - window_days:i]
        if window_data.isna().sum().sum() > 0:
            i += rebalance_every
            continue

        mu = window_data.mean().values
        cov = window_data.cov().values

        # mu = window_data.mean().values
        # cov = window_data.cov().values
        try:
            weights = compute_weights_fn(mu, cov, **kwargs)
        except TypeError:
            weights = compute_weights_fn(window_data, **kwargs)

        if weights is None:
            i += rebalance_every
            continue

        n_assets = returns.shape[1]
        
        if prev_weights is None:
            prev_weights = np.zeros(n_assets)

        if i + rebalance_every < len(dates) - rebalance_every:
            sub_returns = returns.iloc[i:i + rebalance_every]
        else:
            sub_returns = returns.iloc[i:]
            
        for j, (date, row) in enumerate(sub_returns.iterrows()):
            if j == 0 and prev_weights is not None:
                turnover = np.abs(weights - prev_weights).sum()
                tc = turnover * cost
            else:
                tc = 0
            daily_ret = np.dot(weights, row.fillna(0)) - tc
            portfolio_returns.append((date, daily_ret))
            weights_record.append((date, weights))

        prev_weights = weights
        i += rebalance_every

    strat_returns = pd.Series(dict(portfolio_returns)).sort_index()
    # strat_weights = pd.DataFrame({d: w for d, w in weights_record}).T
    

    weights_log = []
    for date, w in weights_record:
        row = {"date": date}
        row.update(dict(zip(returns.columns, w)))
        weights_log.append(row)
    strat_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    strat_returns, strat_weights = convert_to_periodic_returns(strat_returns, strat_weights, period=period)

    return strat_returns, strat_weights


def backtest_daa_from_pivot(
    returns: pd.DataFrame,
    pivot_score: pd.DataFrame,
    window_days: int = 252,
    rebalance_every: int = 20,
    top_n: int = 10,
    score_threshold: float = 0.0,
    cost: float = 0.003,
    period = 20
):
    """
    DAA 전략 20일 리밸런싱 백테스트 함수 (거래비용 포함, 모멘텀 스코어 피벗 사용)

    Parameters:
    - returns: 일간 수익률 DataFrame (index: date, columns: tickers)
    - pivot_score: 모멘텀 스코어 DataFrame (index: date, columns: tickers)
    - window_days: 기록용 (사용 X)
    - rebalance_every: 리밸런싱 주기
    - top_n: 상위 n개 종목 선택
    - score_threshold: 모멘텀 스코어 최소 기준
    - cost: 거래비용

    Returns:
    - daa_returns: 전략 수익률 Series
    - daa_weights: 포트폴리오 비중 DataFrame
    """
    dates = returns.index
    portfolio_returns = []
    weights_record = []
    prev_weights = pd.Series(0, index=returns.columns, dtype=float)

    i = window_days
    while i < len(dates) - rebalance_every+1:
        current_date = dates[i]
        score_today = pivot_score.loc[current_date]

        # 스코어가 0보다 큰 종목 필터링
        satisfied_assets = score_today[score_today > score_threshold]
        if satisfied_assets.empty:
            selected = []
        else:
            selected = satisfied_assets.sort_values(ascending=False).head(top_n).index

        weights = pd.Series(0, index=returns.columns, dtype=float)
        if len(selected) > 0:
            weights[selected] = 1 / len(selected)

        if i + rebalance_every < len(dates) - rebalance_every:
            sub_returns = returns.iloc[i:i + rebalance_every]
        else:
            sub_returns = returns.iloc[i:]
        
        for j, (date, row) in enumerate(sub_returns.iterrows()):
            tc = np.abs(weights - prev_weights).sum() * cost if j == 0 else 0
            daily_ret = np.dot(weights, row.fillna(0)) - tc
            portfolio_returns.append((date, daily_ret))
            weights_record.append((date, weights.copy()))

        prev_weights = weights.copy()
        i += rebalance_every
    

    daa_returns = pd.Series(dict(portfolio_returns)).sort_index()

    weights_log = []
    for date, w in weights_record:
        row = {"date": date}
        row.update(dict(zip(returns.columns, w)))
        weights_log.append(row)
    daa_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    daa_returns, daa_weights = convert_to_periodic_returns(daa_returns, daa_weights, period=period)

    return daa_returns, daa_weights


def backtest_paa_from_pivot(returns: pd.DataFrame,
                             pivot_score: pd.DataFrame,
                             window_days: int = 252,
                             rebalance_every: int = 20,
                             top_n: int = 10,
                             score_threshold: float = 0.0,
                             cost: float = 0.003,
                             period = 20):
    """
    PAA 전략 백테스트 (수익률 기반 점수 사용, 거래비용 반영)

    Parameters:
    - returns: 일간 수익률 DataFrame (index: date, columns: tickers)
    - pivot_score: PAA score DataFrame (index: date, columns: tickers)
    - window_days: 과거 데이터 시작 시점
    - rebalance_every: 리밸런싱 주기
    - top_n: 상위 자산 수
    - score_threshold: 점수 필터링 기준
    - cost: 거래 비용 비율

    Returns:
    - paa_returns: 전략 일간 수익률 Series
    - paa_weights: 전략 리밸런싱 시점별 자산 비중 DataFrame
    """
    dates = returns.index
    portfolio_returns = []
    weights_record = []
    prev_weights = pd.Series(0, index=returns.columns, dtype=float)

    i = window_days
    while i < len(dates) - rebalance_every+1:
        current_date = dates[i]
        score_today = pivot_score.loc[current_date]
        satisfied_assets = score_today[score_today > score_threshold]
        if satisfied_assets.empty:
            selected = []
        else:
            selected = satisfied_assets.sort_values(ascending=False).head(top_n).index

        weights = pd.Series(0, index=returns.columns, dtype=float)
        if len(selected) > 0:
            weights[selected] = 1 / len(selected)

        if i + rebalance_every < len(dates) - rebalance_every:
            sub_returns = returns.iloc[i:i + rebalance_every]
        else:
            sub_returns = returns.iloc[i:]
                    
        for j, (date, row) in enumerate(sub_returns.iterrows()):
            tc = np.abs(weights - prev_weights).sum() * cost if j == 0 else 0
            daily_ret = np.dot(weights, row.fillna(0)) - tc
            portfolio_returns.append((date, daily_ret))
            weights_record.append((date, weights.copy()))

        prev_weights = weights.copy()
        i += rebalance_every

    paa_returns = pd.Series(dict(portfolio_returns)).sort_index()

    weights_log = []
    for date, w in weights_record:
        row = {"date": date}
        row.update(dict(zip(returns.columns, w)))
        weights_log.append(row)
    paa_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    
    paa_returns, paa_weights = convert_to_periodic_returns(paa_returns, paa_weights, period=period)

    return paa_returns, paa_weights


def backtest_gtaa_from_pivot(returns: pd.DataFrame,
                              pivot_close: pd.DataFrame,
                              pivot_sma: pd.DataFrame,
                              window_days: int = 252,
                              rebalance_every: int = 20,
                              top_n: int = 10,
                              cost: float = 0.003):
    """
    GTAA 전략 백테스트 (SMA 기준 + SMA 대비 상대 수익률 정렬, 거래비용 반영)

    Parameters:
    - returns: 일간 수익률 DataFrame (index: date, columns: tickers)
    - pivot_close: 종가 DataFrame (index: date, columns: tickers)
    - pivot_sma: SMA_220 DataFrame (index: date, columns: tickers)
    - window_days: 과거 데이터 시작 시점 (기록용)
    - rebalance_every: 리밸런싱 주기
    - top_n: 상위 자산 수
    - cost: 거래 비용 비율

    Returns:
    - gtaa_returns: 전략 일간 수익률 Series
    - gtaa_weights: 전략 리밸런싱 시점별 자산 비중 DataFrame
    """
    dates = returns.index
    portfolio_returns = []
    weights_record = []
    prev_weights = pd.Series(0, index=returns.columns, dtype=float)

    i = window_days
    while i < len(dates) - rebalance_every+1:
        current_date = dates[i]
        close_today = pivot_close.loc[current_date]
        sma_today = pivot_sma.loc[current_date]

        # 조건: 현재 종가가 SMA보다 큰 자산
        condition = (close_today > sma_today)
        filtered_assets = close_today[condition]

        # 정렬 기준: SMA 대비 수익률 비율 (close / sma - 1)
        relative_returns = (filtered_assets / sma_today[condition]) - 1
        selected = relative_returns.sort_values(ascending=False).head(top_n).index if not relative_returns.empty else []

        # 포트폴리오 비중 계산
        weights = pd.Series(0, index=returns.columns, dtype=float)
        if len(selected) > 0:
            weights[selected] = 1 / len(selected)
        else:
            selected = []

        # 수익률 계산
        if i + rebalance_every < len(dates) - rebalance_every:
            sub_returns = returns.iloc[i:i + rebalance_every]
        else:
            sub_returns = returns.iloc[i:]
                    
        for j, (date, row) in enumerate(sub_returns.iterrows()):
            tc = np.abs(weights - prev_weights).sum() * cost if j == 0 else 0
            daily_ret = np.dot(weights, row.fillna(0)) - tc
            portfolio_returns.append((date, daily_ret))
            weights_record.append((date, weights.copy()))

        prev_weights = weights.copy()
        i += rebalance_every

    gtaa_returns = pd.Series(dict(portfolio_returns)).sort_index()
    weights_log = []
    for date, w in weights_record:
        row = {"date": date}
        row.update(dict(zip(returns.columns, w)))
        weights_log.append(row)
    gtaa_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    
    return gtaa_returns, gtaa_weights


def backtest_equal_weight_20day(returns: pd.DataFrame, rebalance_every=20, window_days=252, cost=0.003, period = 20):
    dates = returns.index
    portfolio_returns = []
    weights_record = []
    prev_weights = None

    i = window_days
    while i < len(dates) - rebalance_every+1:
        rebalance_day = dates[i]
        window_data = returns.iloc[i - window_days:i]


        n_assets = window_data.shape[1]
        weights = np.ones(n_assets) / n_assets  # 1/N 포트폴리오
        n_assets = returns.shape[1]
        if prev_weights is None:
            prev_weights = np.zeros(n_assets)  # 최초 리밸런싱: 전량 매수로 간주
            
        if i + rebalance_every < len(dates) - rebalance_every:
            sub_returns = returns.iloc[i:i + rebalance_every]
        else:
            sub_returns = returns.iloc[i:]

        for j, (date, row) in enumerate(sub_returns.iterrows()):
            if j == 0 and prev_weights is not None:
                turnover = np.abs(weights - prev_weights).sum()
                tc = turnover * cost
            else:
                tc = 0
            daily_ret = np.dot(weights, row.fillna(0)) - tc
            portfolio_returns.append((date, daily_ret))
            weights_record.append((date, weights))

        prev_weights = weights
        i += rebalance_every

    ew_returns = pd.Series(dict(portfolio_returns)).sort_index()

    weights_log = []
    for date, w in weights_record:
        row = {"date": date}
        row.update(dict(zip(returns.columns, w)))
        weights_log.append(row)
    ew_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    ew_returns, ew_weights = convert_to_periodic_returns(ew_returns, ew_weights, period=period)

    return ew_returns, ew_weights








# def backtest_lowvol_20day_noshort(returns: pd.DataFrame,
#                                    window_days=252,
#                                    rebalance_every=20,
#                                    top_pct=0.1,
#                                    cost=0.003):
#     """
#     변동성 기준 Low-Risk 포트폴리오 전략
#     - 과거 252일 기준 낮은 변동성 종목 선택 (top_pct 비중)
#     - 20일마다 리밸런싱
#     - 거래비용 포함

#     Parameters:
#     - returns: 종가 DataFrame (index: date, columns: ticker)
#     - window_days: 변동성 계산에 사용할 기간
#     - rebalance_every: 리밸런싱 주기 (일)
#     - top_pct: 상위 몇 %를 low volatility로 볼지 (예: 0.1 = 10%)
#     - cost: 거래비용

#     Returns:
#     - lowvol_returns: 전략 수익률 Series
#     - lowvol_weights: 리밸런싱 시점별 포트폴리오 비중 DataFrame
#     """
#     dates = returns.index
#     portfolio_returns = []
#     weights_record = []
#     prev_weights = pd.Series(0, index=returns.columns)

#     i = window_days
#     while i < len(dates) - rebalance_every+1:
#         window_data = returns.iloc[i - window_days:i]
#         current_vol = window_data.std()
#         selected = current_vol.nsmallest(int(len(current_vol) * top_pct)).index

#         weights = pd.Series(0, index=returns.columns, dtype=float)
#         weights[selected] = 1 / len(selected)

#         if i + rebalance_every < len(dates) - rebalance_every:
#             sub_returns = returns.iloc[i:i + rebalance_every]
#         else:
#             sub_returns = returns.iloc[i:]
            
#         for j, (date, row) in enumerate(sub_returns.iterrows()):
#             if j == 0:
#                 turnover = np.abs(weights - prev_weights).sum()
#                 tc = turnover * cost
#             else:
#                 tc = 0
#             daily_ret = np.dot(weights, row.fillna(0)) - tc
#             portfolio_returns.append((date, daily_ret))
#             weights_record.append((date, weights.copy()))
#         prev_weights = weights
#         i += rebalance_every

#     lowvol_returns = pd.Series(dict(portfolio_returns)).sort_index()
#     weights_log = []
#     for date, w in weights_record:
#         row = {"date": date}
#         row.update(dict(zip(returns.columns, w)))
#         weights_log.append(row)
#     lowvol_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    
#     return lowvol_returns, lowvol_weights


# def backtest_dual_momentum_from_pivot(
#     returns: pd.DataFrame,
#     pivot_momentum: pd.DataFrame,
#     window_days: int = 252,
#     rebalance_every: int = 20,
#     top_n: int = 10,
#     risk_free_threshold: float = 1.0,
#     cost: float = 0.003
# ):
#     """
#     듀얼 모멘텀 전략 20일 리밸런싱 백테스트 함수 (거래비용 포함, 모멘텀 피벗 사용)

#     Parameters:
#     - returns: 일간 수익률 DataFrame (index: date, columns: tickers)
#     - pivot_momentum: 모멘텀 지표 DataFrame (누적 수익률 or 점수 등)
#     - window_days: 모멘텀 유효성을 평가하기 위한 lookback 기간 (기록용)
#     - rebalance_every: 리밸런싱 주기 (기본 20일)
#     - top_n: 상위 n개의 자산에 균등 투자
#     - risk_free_threshold: 모멘텀 기준 (1.0이면 break-even 수익률 이상)
#     - cost: 거래비용 비율

#     Returns:
#     - dm_returns: 전략 수익률 Series
#     - dm_weights: 포트폴리오 비중 DataFrame
#     """
#     dates = returns.index
#     portfolio_returns = []
#     weights_record = []
#     prev_weights = pd.Series(0, index=returns.columns, dtype=float)

#     i = window_days
#     while i < len(dates) - rebalance_every+1:
#         current_date = dates[i]
#         momentum_window = pivot_momentum.loc[current_date]

#         # risk_free_threshold 만족 종목 필터링
#         satisfied_assets = momentum_window[momentum_window > risk_free_threshold]
#         if satisfied_assets.empty:
#             selected = []
#         else:
#             selected = satisfied_assets.sort_values(ascending=False).head(top_n).index

#         weights = pd.Series(0, index=returns.columns, dtype=float)
#         if len(selected) > 0:
#             weights[selected] = 1 / len(selected)

#         if i + rebalance_every < len(dates) - rebalance_every:
#             sub_returns = returns.iloc[i:i + rebalance_every]
#         else:
#             sub_returns = returns.iloc[i:]
            
#         for j, (date, row) in enumerate(sub_returns.iterrows()):
#             tc = np.abs(weights - prev_weights).sum() * cost if j == 0 else 0
#             daily_ret = np.dot(weights, row.fillna(0)) - tc
#             portfolio_returns.append((date, daily_ret))
#             weights_record.append((date, weights.copy()))

#         prev_weights = weights.copy()
#         i += rebalance_every

#     dm_returns = pd.Series(dict(portfolio_returns)).sort_index()
#     weights_log = []
#     for date, w in weights_record:
#         row = {"date": date}
#         row.update(dict(zip(returns.columns, w)))
#         weights_log.append(row)
#     dm_weights = pd.DataFrame(weights_log).set_index("date").sort_index()
    
#     return dm_returns, dm_weights
