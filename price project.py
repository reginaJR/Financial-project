"""
European Option Pricing and Greeks Analysis
使用最近6个月历史数据校准模型，进行欧式看涨期权定价与 Greeks 分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

# ------------------------------
# 1. 数据获取：最近6个月日线数据
# ------------------------------
import akshare as ak

# 获取沪深300 ETF（510300）最近6个月日线数据
symbol = "510300"
start_date = (datetime.today() - timedelta(days=180)).strftime("%Y%m%d")
end_date = datetime.today().strftime("%Y%m%d")

# ak.fund_etf_hist_em 用于获取 ETF 历史数据
df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
prices = df['收盘'].dropna()
prices.index = pd.to_datetime(df['日期'])

# ------------------------------
# 2. 参数估计
# ------------------------------
# 计算日收益率
returns = prices.pct_change().dropna()

# 年化波动率（使用样本标准差，假设252个交易日）
sigma_daily = returns.std()
sigma = sigma_daily * np.sqrt(252)

# 年化收益率（仅作参考，风险中性定价中不使用）
mu_daily = returns.mean()
mu = mu_daily * 252

# 无风险利率（可使用当前短期利率，这里设定为3%）
r = 0.03

# 当前价格（使用最后一个收盘价）
S0 = prices.iloc[-1]

print(f"\n--- Estimated Parameters ---")
print(f"Current price S0: {S0:.2f}")
print(f"Annualized volatility (sigma): {sigma:.2%}")
print(f"Annualized drift (mu, for reference): {mu:.2%}")
print(f"Risk-free rate r: {r:.2%}")

# ------------------------------
# 3. 蒙特卡洛模拟定价函数
# ------------------------------
def monte_carlo_call_price(S0, K, T, r, sigma, M=10000, N=252, seed=42):
    """
    使用几何布朗运动模拟标的资产价格路径，计算欧式看涨期权价格。
    S0: 当前价格
    K: 执行价
    T: 到期时间（年）
    r: 无风险利率
    sigma: 年化波动率
    M: 模拟路径数
    N: 时间步数
    seed: 随机种子，保证结果可复现
    """
    np.random.seed(seed)
    dt = T / N
    S = np.zeros((M, N+1))
    S[:, 0] = S0

    # 几何布朗运动：S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    for t in range(1, N+1):
        z = np.random.standard_normal(M)
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # 到期时的看涨期权收益
    payoff = np.maximum(S[:, -1] - K, 0)
    # 折现求期望
    call_price = np.exp(-r * T) * np.mean(payoff)
    return call_price, S   # 返回价格和路径，用于后续可视化

# ------------------------------
# 4. Black-Scholes 解析解定价与 Greeks
# ------------------------------
def black_scholes_call(S0, K, T, r, sigma):
    """
    计算欧式看涨期权价格及 Greeks
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 期权价格
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Greeks
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100   # 除以100，表示波动率每变动1%
    theta = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # 除以365，表示每天的时间损耗
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # 除以100，表示利率每变动1%

    return call_price, delta, gamma, vega, theta, rho

# ------------------------------
# 5. 运行定价与 Greeks
# ------------------------------
T = 1.0                     # 1年到期
K = S0 * 1.05               # 执行价设为当前价格的105%
M = 10000                   # 模拟路径数
N = 252                     # 时间步数

print(f"\n--- Option Contract ---")
print(f"Strike K: {K:.2f}")
print(f"Time to maturity T: {T} year(s)")
print(f"Number of Monte Carlo paths: {M}")

# 蒙特卡洛定价
mc_price, sim_paths = monte_carlo_call_price(S0, K, T, r, sigma, M, N)

# Black-Scholes 定价与 Greeks
bs_price, delta, gamma, vega, theta, rho = black_scholes_call(S0, K, T, r, sigma)

print(f"\n--- Pricing Results ---")
print(f"Monte Carlo Call Price: {mc_price:.4f}")
print(f"Black-Scholes Call Price: {bs_price:.4f}")
print(f"Difference: {abs(mc_price - bs_price):.6f} ({(abs(mc_price - bs_price)/bs_price)*100:.2f}% of BS)")

print(f"\n--- Greeks (Black-Scholes) ---")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"Vega (per 1% vol change): {vega:.4f}")
print(f"Theta (per day): {theta:.6f}")
print(f"Rho (per 1% rate change): {rho:.6f}")

# ------------------------------
# 6. 可视化
# ------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：模拟路径示例（前20条）
ax1 = axes[0, 0]
ax1.plot(sim_paths[:20, :].T, linewidth=0.8, alpha=0.7)
ax1.axhline(y=K, color='red', linestyle='--', label=f'Strike K={K:.2f}')
ax1.set_title(f'Simulated Price Paths ({M} paths, {N} steps)')
ax1.set_xlabel('Time step (days)')
ax1.set_ylabel('Price')
ax1.legend()

# 图2：期权价格随标的资产价格变化（MC vs BS）
ax2 = axes[0, 1]
S_range = np.linspace(S0*0.8, S0*1.2, 20)
mc_prices = []
bs_prices = []
for S in S_range:
    mc_p, _ = monte_carlo_call_price(S, K, T, r, sigma, M=5000, N=252, seed=1)
    mc_prices.append(mc_p)
    bs_p, _, _, _, _, _ = black_scholes_call(S, K, T, r, sigma)
    bs_prices.append(bs_p)
ax2.plot(S_range, bs_prices, 'b-', label='Black-Scholes')
ax2.plot(S_range, mc_prices, 'ro', markersize=4, label='Monte Carlo')
ax2.axvline(x=S0, color='green', linestyle=':', label=f'Current S0={S0:.2f}')
ax2.set_title('Option Price vs Underlying Price')
ax2.set_xlabel('Underlying Price S')
ax2.set_ylabel('Call Option Price')
ax2.legend()

# 图3：Greeks 随标的价格变化（仅 Delta, Gamma, Vega）
ax3 = axes[1, 0]
S_range_fine = np.linspace(S0*0.8, S0*1.2, 50)
deltas = []
gammas = []
vegas = []
for S in S_range_fine:
    _, d, g, v, _, _ = black_scholes_call(S, K, T, r, sigma)
    deltas.append(d)
    gammas.append(g)
    vegas.append(v)
ax3.plot(S_range_fine, deltas, label='Delta')
ax3.plot(S_range_fine, gammas, label='Gamma')
ax3.plot(S_range_fine, vegas, label='Vega (scaled)')
ax3.set_title('Greeks vs Underlying Price')
ax3.set_xlabel('Underlying Price S')
ax3.set_ylabel('Greek Value')
ax3.legend()

# 图4：到期收益分布直方图
ax4 = axes[1, 1]
terminal_prices = sim_paths[:, -1]
ax4.hist(terminal_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax4.axvline(x=K, color='red', linestyle='--', label=f'Strike K={K:.2f}')
ax4.set_title('Distribution of Terminal Prices (Monte Carlo)')
ax4.set_xlabel('Price at Maturity')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
plt.savefig('option_pricing_greeks_analysis.png', dpi=150)
plt.show()

print("\n图表已保存为 option_pricing_greeks_analysis.png")
print("项目完成！")