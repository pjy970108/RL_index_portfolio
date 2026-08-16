from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd


Alternative = Literal["greater", "less", "two-sided"]


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    estimate: float
    standard_error: float
    n_obs: int


@dataclass(frozen=True)
class SharpeDiffResult:
    model_sharpe: float
    benchmark_sharpe: float
    sharpe_difference: float
    p_value: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_obs: int


def _clean_series(returns: pd.Series | Iterable[float]) -> pd.Series:
    series = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("returns must contain at least one finite observation")
    return series


def _alternative_p_value(statistic: float, df: int, alternative: Alternative) -> float:
    _, t_dist = _load_scipy_stats()
    if alternative == "greater":
        return float(1.0 - t_dist.cdf(statistic, df=df))
    if alternative == "less":
        return float(t_dist.cdf(statistic, df=df))
    if alternative == "two-sided":
        return float(2.0 * (1.0 - t_dist.cdf(abs(statistic), df=df)))
    raise ValueError(f"unsupported alternative: {alternative}")


def _load_scipy_stats():
    try:
        from scipy.stats import norm, t as t_dist
    except ImportError as exc:
        raise ImportError(
            "scipy is required for statistical-test p-values. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return norm, t_dist


def _load_statsmodels():
    try:
        import statsmodels.api as sm
        from statsmodels.tsa.ar_model import AutoReg
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for Newey-West and Ledoit-Wolf HAC tests. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return sm, AutoReg


def newey_west_mean_test(
    returns: pd.Series | Iterable[float],
    *,
    lags: int = 12,
    alternative: Alternative = "greater",
) -> TestResult:
    """Test whether the mean return differs from zero with a Newey-West HAC SE.

    The thesis evaluation uses monthly test-period returns. The default
    alternative is positive mean return, matching the retained analysis
    notebooks where p-values were computed as ``1 - CDF(t_stat)``.
    """

    series = _clean_series(returns)
    sm, _ = _load_statsmodels()
    y = series.to_numpy()
    x = np.ones((len(y), 1), dtype="float64")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    estimate = float(model.params[0])
    standard_error = float(model.bse[0])
    statistic = estimate / standard_error if standard_error else np.nan
    p_value = _alternative_p_value(float(statistic), df=max(len(y) - 1, 1), alternative=alternative)

    return TestResult(
        statistic=float(statistic),
        p_value=p_value,
        estimate=estimate,
        standard_error=standard_error,
        n_obs=len(y),
    )


def annualized_return(returns: pd.Series | Iterable[float], periods_per_year: int = 12) -> float:
    series = _clean_series(returns)
    cumulative = float(np.prod(1.0 + series.to_numpy()))
    return cumulative ** (periods_per_year / len(series)) - 1.0


def max_drawdown(returns: pd.Series | Iterable[float]) -> float:
    series = _clean_series(returns)
    wealth = np.cumprod(1.0 + series.to_numpy())
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    return float(abs(np.min(drawdown)))


def profit_factor(returns: pd.Series | Iterable[float]) -> float:
    series = _clean_series(returns)
    gross_profit = float(series[series > 0].sum())
    gross_loss = float(series[series < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return gross_profit / abs(gross_loss)


def performance_metrics(
    returns: pd.Series | Iterable[float],
    *,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    series = _clean_series(returns)
    values = series.to_numpy()
    ann_return = annualized_return(series, periods_per_year=periods_per_year)
    volatility = float(np.std(values, ddof=1) * np.sqrt(periods_per_year))
    downside = values[values < 0]
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    mdd = max_drawdown(series)

    return {
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe": (ann_return - risk_free_rate) / volatility if volatility else 0.0,
        "sortino": (ann_return - risk_free_rate) / downside_vol if downside_vol else 0.0,
        "calmar": ann_return / mdd if mdd else 0.0,
        "mdd": mdd,
        "profit_factor": profit_factor(series),
    }


def monthly_distribution_summary(returns: pd.Series | Iterable[float]) -> dict[str, float]:
    series = _clean_series(returns)
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
    }


def low_quantile_summary(returns: pd.Series | Iterable[float], quantile: float = 0.25) -> dict[str, float]:
    series = _clean_series(returns)
    threshold = float(series.quantile(quantile))
    bottom = series[series <= threshold]
    return {
        "threshold": threshold,
        "mean": float(bottom.mean()),
        "std": float(bottom.std(ddof=1)),
        "min": float(bottom.min()),
        "max": float(bottom.max()),
        "skewness": float(bottom.skew()),
        "kurtosis": float(bottom.kurtosis()),
        "n_obs": float(len(bottom)),
    }


def profit_loss_summary(returns: pd.Series | Iterable[float]) -> dict[str, float]:
    series = _clean_series(returns)
    gains = series[series > 0]
    losses = series[series < 0]
    return {
        "profit_months": float(len(gains)),
        "loss_months": float(len(losses)),
        "profit_mean": float(gains.mean()) if len(gains) else 0.0,
        "loss_mean": float(losses.mean()) if len(losses) else 0.0,
        "profit_std": float(gains.std(ddof=1)) if len(gains) > 1 else 0.0,
        "loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
    }


def regime_metrics(
    returns: pd.Series,
    regimes: dict[str, tuple[str, str]] | None = None,
    *,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    series = _clean_series(returns)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("returns must use a DatetimeIndex for regime analysis")

    regime_map = regimes or {
        "Pre-Pandemic": ("2019-01-01", "2020-02-29"),
        "Pandemic / Inflation Surge": ("2020-03-01", "2022-06-30"),
        "Global Disinflation": ("2022-07-01", "2024-12-31"),
    }

    rows = {}
    for name, (start, end) in regime_map.items():
        period_returns = series.loc[start:end]
        rows[name] = performance_metrics(period_returns, periods_per_year=periods_per_year)
    return pd.DataFrame.from_dict(rows, orient="index")


def ledoit_wolf_sharpe_diff_test(
    model_returns: pd.Series | Iterable[float],
    benchmark_returns: pd.Series | Iterable[float],
    *,
    alpha: float = 0.05,
) -> SharpeDiffResult:
    """Ledoit-Wolf style HAC test for a model-vs-benchmark Sharpe difference.

    Returns are aligned as ``benchmark, model`` so a positive difference means
    the model Sharpe ratio is higher than the benchmark Sharpe ratio.
    """

    model = _clean_series(model_returns)
    benchmark = _clean_series(benchmark_returns)
    aligned = pd.concat([benchmark, model], axis=1, join="inner").dropna()
    if len(aligned) < 8:
        raise ValueError("at least 8 aligned observations are required")

    norm, _ = _load_scipy_stats()
    values = aligned.to_numpy(dtype="float64")
    sharpe = values.mean(axis=0) / values.std(axis=0, ddof=0)
    standard_error = _relative_sharpe_standard_error(values)
    diff = float(sharpe[1] - sharpe[0])
    p_value = float(2.0 * norm.cdf(-abs(diff) / standard_error))
    half_width = float(norm.ppf(1.0 - alpha / 2.0) * standard_error)

    return SharpeDiffResult(
        model_sharpe=float(sharpe[1]),
        benchmark_sharpe=float(sharpe[0]),
        sharpe_difference=diff,
        p_value=p_value,
        standard_error=float(standard_error),
        confidence_interval=(diff - half_width, diff + half_width),
        n_obs=len(aligned),
    )


def _relative_sharpe_standard_error(returns: np.ndarray) -> float:
    mean = returns.mean(axis=0)
    squared = np.square(returns)
    second_moment = squared.mean(axis=0)
    gradient = np.zeros(4, dtype="float64")
    gradient[0] = second_moment[0] / np.power(second_moment[0] - mean[0] ** 2, 1.5)
    gradient[1] = -second_moment[1] / np.power(second_moment[1] - mean[1] ** 2, 1.5)
    gradient[2] = -0.5 * mean[0] / np.power(second_moment[0] - mean[0] ** 2, 1.5)
    gradient[3] = -0.5 * mean[1] / np.power(second_moment[1] - mean[1] ** 2, 1.5)
    v_hat = np.column_stack(
        [
            returns[:, 0] - mean[0],
            returns[:, 1] - mean[1],
            squared[:, 0] - second_moment[0],
            squared[:, 1] - second_moment[1],
        ]
    )
    psi_hat = _psi_hat(v_hat)
    variance = float(gradient.T @ psi_hat @ gradient / len(returns))
    return float(np.sqrt(max(variance, 0.0)))


def _psi_hat(v_hat: np.ndarray) -> np.ndarray:
    n_obs = len(v_hat)
    bandwidth = min(2.6614 * (_alpha_hat(v_hat) * n_obs) ** 0.2, n_obs - 1)
    psi = _gamma_hat(v_hat, 0)
    lag = 1
    while lag < bandwidth:
        gamma = _gamma_hat(v_hat, lag)
        psi = psi + _parzen_kernel(lag / bandwidth) * (gamma + gamma.T)
        lag += 1
    return (n_obs / (n_obs - v_hat.shape[1])) * psi


def _gamma_hat(v_hat: np.ndarray, lag: int) -> np.ndarray:
    if lag >= len(v_hat):
        raise ValueError("lag must be smaller than the number of observations")
    gamma = np.zeros((v_hat.shape[1], v_hat.shape[1]), dtype="float64")
    for i in range(lag, len(v_hat)):
        gamma += np.outer(v_hat[i], v_hat[i - lag])
    return gamma / len(v_hat)


def _alpha_hat(v_hat: np.ndarray) -> float:
    _, AutoReg = _load_statsmodels()
    numerator = 0.0
    denominator = 0.0
    for col in range(v_hat.shape[1]):
        fit = AutoReg(v_hat[:, col], lags=1, trend="c", old_names=False).fit()
        rho = fit.params[1]
        sigma = np.sqrt(fit.sigma2)
        numerator += 4.0 * (rho**2) * (sigma**4) / ((1.0 - rho) ** 8)
        denominator += (sigma**4) / ((1.0 - rho) ** 4)
    if denominator == 0:
        return 1.0
    return float(max(numerator / denominator, 1e-12))


def _parzen_kernel(x: float) -> float:
    abs_x = abs(x)
    if abs_x <= 0.5:
        return 1.0 - 6.0 * (x**2) + 6.0 * (abs_x**3)
    if abs_x <= 1.0:
        return 2.0 * ((1.0 - abs_x) ** 3)
    return 0.0


def summarize_return_table(
    monthly_returns: pd.DataFrame,
    *,
    model_column: str = "GRPO",
    benchmark_columns: Iterable[str] = (),
    newey_west_lags: int = 12,
) -> dict[str, pd.DataFrame]:
    """Build the final monthly OOS evaluation tables from return columns."""

    returns = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    metrics = pd.DataFrame(
        {
            column: performance_metrics(returns[column].dropna(), periods_per_year=12)
            for column in returns.columns
        }
    ).T
    mean_tests = pd.DataFrame(
        {
            column: newey_west_mean_test(returns[column].dropna(), lags=newey_west_lags).__dict__
            for column in returns.columns
        }
    ).T
    sharpe_tests = pd.DataFrame(
        {
            benchmark: ledoit_wolf_sharpe_diff_test(
                returns[model_column],
                returns[benchmark],
            ).__dict__
            for benchmark in benchmark_columns
        }
    ).T
    return {
        "performance": metrics,
        "newey_west_mean_tests": mean_tests,
        "ledoit_wolf_sharpe_tests": sharpe_tests,
    }
