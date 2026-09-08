# -*- coding: utf-8 -*-
"""
Portfolio Market Risk Analysis: VaR, CVaR and GARCH Volatility Modeling
数据源：腾讯财经公开接口
主分析区间：2019-08-01 至 2026-08-01（全样本，约1697个交易日）
压力测试区间：2020-02-20 至 2020-03-23（COVID-19 全球市场崩盘期）

本次更新（相对初版）：
1. 压力测试新增：压力期交易日数、组合累计收益率（精确值，替代推算）、单日最差收益，
   并与正常期 99% VaR 对比
2. 蒙特卡洛函数修复 days>1 时的均值复利问题，并新增 Student-t 扰动选项（厚尾验证）
3. GARCH 增加正态 vs t 分布对比（按 AIC 选择），动态 VaR 基于更优模型
4. 新增 Kupiec POF 回测（VaR 突破次数检验，样本内）
5. 图表标题口径修正为 2019-2026（初版误写 2024.08-2026.08）
6. 图4 累计收益曲线改为以压力窗口起点归一
7. 组合统计新增夏普比率
"""

import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, t
from arch import arch_model

# ------------------------------
# 全局设置
# ------------------------------
FULL_START_DATE = '20190801'
FULL_END_DATE = '20260801'

ANALYSIS_START = '2019-08-01'
ANALYSIS_END = '2026-08-01'

SYMBOLS = {
    'stock': '510300',   # 沪深300 ETF
    'bond': '511010',    # 国债 ETF
    'gold': '518880'     # 黄金 ETF
}

RISK_FREE_RATE = 0.02    # 年化无风险利率，用于夏普比率，可按需调整

# ------------------------------
# 数据获取函数：腾讯财经接口（主）+ AKShare（兜底）
# ------------------------------
def get_etf_data(symbol, name, start_date, end_date, retries=3):
    """
    获取 ETF 日线收盘价（前复权），数据源：腾讯财经 ifzq.gtimg.cn。
    - 单次请求上限 800 条，7 年数据自动分页向前翻取；
    - 字段顺序：[日期, 开盘, 收盘, 最高, 最低, 成交量]，取 index 2 收盘价；
    - AKShare（fund_etf_hist_em）作为兜底通道。
    symbol: 6位代码，如 510300（沪市），159919（深市）
    """
    full_code = ('sh' if symbol.startswith(('5', '6')) else 'sz') + symbol
    # 日期规范化：腾讯接口要求 YYYY-MM-DD 格式（东财式的 20190801 会触发 param error）
    sd = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    ed = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://gu.qq.com/",
    }

    def _request_chunk(s, e):
        """请求 [s, e] 区间内最近 800 条前复权日K，返回 [[date, close], ...]"""
        param = f"{full_code},day,{s},{e},800,qfq"
        for attempt in range(retries):
            try:
                resp = requests.get(url, params={"param": param}, headers=headers, timeout=15)
                d = resp.json()
                data = d.get('data')
                # 注意：参数错误时接口也返回 code=0，但 data 是列表而非字典——必须显式校验
                if not isinstance(data, dict) or full_code not in data:
                    raise ValueError(f"接口返回异常: code={d.get('code')}, msg={d.get('msg')}")
                node = data[full_code]
                klines = node.get('qfqday') or node.get('day') or []
                return [[row[0], float(row[2])] for row in klines]
            except Exception as e:
                print(f"✗ {name} ({symbol}) 区间 {s}~{e} 第 {attempt+1}/{retries} 次失败: {e}")
                if attempt < retries - 1:
                    time.sleep(3)
        return []

    # ---------- 方式①：腾讯财经（自动分页，最多回翻 5 页） ----------
    all_rows, cur_end, page = [], ed, 0
    while page < 5:
        chunk = _request_chunk(sd, cur_end)
        if not chunk:
            break
        all_rows = chunk + all_rows           # chunk 是较早的数据，拼在前面
        first_date = chunk[0][0]
        page += 1
        if first_date <= sd or len(chunk) < 800:
            break                             # 已覆盖到起点（或区间内不足 800 条）
        cur_end = (pd.to_datetime(first_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        time.sleep(1)

    if all_rows:
        df = pd.DataFrame(all_rows, columns=['date', 'close']).drop_duplicates('date')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        df = df.loc[sd:ed]
        print(f"✓ 成功获取 {name} ({symbol}) [腾讯财经，{page} 页]，共 {len(df)} 条数据")
        return df[['close']].rename(columns={'close': name})

    # ---------- 方式②：AKShare 兜底 ----------
    try:
        import akshare as ak
        df_ak = ak.fund_etf_hist_em(symbol=symbol, period="daily",
                                    start_date=start_date.replace('-', ''),
                                    end_date=end_date.replace('-', ''),
                                    adjust="qfq")
        df = df_ak.rename(columns={"日期": "date", "收盘": "close"})[['date', 'close']].copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print(f"✓ 成功获取 {name} ({symbol}) [AKShare 兜底]，共 {len(df)} 条数据")
        return df[['close']].rename(columns={'close': name})
    except Exception as e:
        print(f"✗ AKShare 兜底也失败: {e}")

    raise Exception(
        f"无法获取 {name} ({symbol}) 的数据。请检查：\n"
        f"  1) 浏览器能否打开 https://gu.qq.com/sh{symbol} （验证网络可达腾讯财经）；\n"
        f"  2) 若使用 VPN/代理，请关闭后重试；\n"
        f"  3) 或运行 pip install akshare 后重试（脚本会自动走兜底通道）。"
    )

# ------------------------------
# VaR / CVaR 计算函数
# ------------------------------
def historical_var_cvar(r, confidence):
    var = np.percentile(r, (1 - confidence) * 100)
    cvar = r[r <= var].mean()
    return var, cvar

def parametric_var_cvar(r, confidence):
    mu = r.mean()
    sigma = r.std()
    var = mu + norm.ppf(1 - confidence) * sigma
    cvar = mu - sigma * norm.pdf(norm.ppf(confidence)) / (1 - confidence)
    return var, cvar

def monte_carlo_var_cvar(r, confidence, n_sim=10000, seed=42, days=1, dist='normal', df=5):
    """
    蒙特卡洛模拟 VaR/CVaR。
    days=1 时结果与初版一致（保持随机种子不变以复现原结果）。
    dist='t'：使用标准化 Student-t 扰动（自由度 df），用于厚尾情景验证——
    与参数法不再共享正态假设，构成真正独立的方法验证。
    修复点：多日收益通过路径连乘 (1+r) 累积，均值与波动率的复利处理均正确。
    """
    mu = r.mean()
    sigma = r.std()
    np.random.seed(seed)
    if dist == 'normal':
        shocks = np.random.normal(0, 1, size=(n_sim, days))
    else:
        z = np.random.standard_t(df, size=(n_sim, days))
        z = z / np.sqrt(df / (df - 2))   # 标准化为单位方差
        shocks = z
    sim_paths = mu + sigma * shocks
    cum = np.prod(1 + sim_paths, axis=1) - 1
    var = np.percentile(cum, (1 - confidence) * 100)
    cvar = cum[cum <= var].mean()
    return var, cvar

def kupiec_pof(exceptions, n, confidence):
    """
    Kupiec POF（Proportion of Failures）回测。
    H0：VaR 突破频率 = 理论值 (1 - confidence)。
    返回 LR 统计量与 p 值；p > 0.05 表示不能拒绝模型（模型合格）。
    """
    x = int(exceptions)
    p0 = 1 - confidence
    p_hat = x / n
    if x == 0:
        lr = -2 * (n * np.log(1 - p0))
    else:
        log_null = (n - x) * np.log(1 - p0) + x * np.log(p0)
        log_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        lr = -2 * (log_null - log_alt)
    p_value = 1 - chi2.cdf(lr, df=1)
    return lr, p_value

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    print("正在下载完整历史数据...")
    try:
        df_stock = get_etf_data(SYMBOLS['stock'], 'stock', FULL_START_DATE, FULL_END_DATE)
        df_bond = get_etf_data(SYMBOLS['bond'], 'bond', FULL_START_DATE, FULL_END_DATE)
        df_gold = get_etf_data(SYMBOLS['gold'], 'gold', FULL_START_DATE, FULL_END_DATE)
    except Exception as e:
        print(str(e))
        print("请尝试切换网络环境或稍后重试。")
        exit(1)

    prices_full = pd.concat([df_stock, df_bond, df_gold], axis=1).dropna()
    print(f"\n完整数据范围: {prices_full.index[0].date()} 至 {prices_full.index[-1].date()}")
    print(f"完整数据交易日数: {len(prices_full)}")

    prices = prices_full.loc[ANALYSIS_START:ANALYSIS_END].dropna()
    print(f"主分析数据范围: {prices.index[0].date()} 至 {prices.index[-1].date()}")
    print(f"主分析交易日数: {len(prices)}")

    # ------------------------------
    # 收益率与组合统计
    # ------------------------------
    returns = prices.pct_change().dropna()
    weights = np.array([0.5, 0.3, 0.2])
    port_returns = returns.dot(weights)

    print("\n--- 主分析区间组合年化收益率与波动率 ---")
    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
    print(f"年化收益率: {ann_return:.2%}")
    print(f"年化波动率: {ann_vol:.2%}")
    print(f"夏普比率 (rf={RISK_FREE_RATE:.0%}): {sharpe:.2f}")

    returns_full = prices_full.pct_change().dropna()
    port_returns_full = returns_full.dot(weights)

    # ------------------------------
    # 主分析区间 1-day VaR / CVaR：三种方法（含 t 分布 MC 对比）
    # ------------------------------
    confidence_levels = [0.95, 0.99]

    print("\n--- 主分析区间 1-day VaR 和 CVaR ---")
    for cl in confidence_levels:
        hist_var, hist_cvar = historical_var_cvar(port_returns, cl)
        param_var, param_cvar = parametric_var_cvar(port_returns, cl)
        mc_var, mc_cvar = monte_carlo_var_cvar(port_returns, cl, days=1, dist='normal')
        mct_var, mct_cvar = monte_carlo_var_cvar(port_returns, cl, days=1, dist='t', df=5)
        print(f"\n置信水平 {cl:.0%}:")
        print(f"  历史模拟法:       VaR = {hist_var:.4f}, CVaR = {hist_cvar:.4f}")
        print(f"  参数法:           VaR = {param_var:.4f}, CVaR = {param_cvar:.4f}")
        print(f"  蒙特卡洛(正态):   VaR = {mc_var:.4f}, CVaR = {mc_cvar:.4f}")
        print(f"  蒙特卡洛(t,df=5): VaR = {mct_var:.4f}, CVaR = {mct_cvar:.4f}")

    print("\n解读：正态 MC 与参数法结果接近（共享正态假设，非独立验证）；"
          "t 分布 MC 的 CVaR 若向历史模拟值靠拢，则独立印证了厚尾风险。")

    # ------------------------------
    # GARCH(1,1)：正态 vs t 分布对比，按 AIC 选择
    # ------------------------------
    port_returns_pct = port_returns * 100
    res_n = arch_model(port_returns_pct, vol='Garch', p=1, q=1, mean='Constant', dist='normal').fit(disp='off')
    res_t = arch_model(port_returns_pct, vol='Garch', p=1, q=1, mean='Constant', dist='t').fit(disp='off')

    print("\n--- GARCH(1,1) 分布选择（正态 vs t） ---")
    print(f"正态: LogLik = {res_n.loglikelihood:.2f}, AIC = {res_n.aic:.2f}")
    print(f"t:    LogLik = {res_t.loglikelihood:.2f}, AIC = {res_t.aic:.2f}")
    if res_t.aic < res_n.aic:
        res = res_t
        print("→ t 分布更优（AIC 更低），采用 t 分布模型")
        print(f"  标准化残差自由度 nu = {res.params['nu']:.2f}（越小尾部越厚）")
    else:
        res = res_n
        print("→ 正态分布更优（AIC 更低），采用正态分布模型")

    # 分布一致性：所选分布对应的标准化下尾分位数函数
    # （t 模型若仍用 norm.ppf 取 VaR，会低估尾部——初版代码的遗留问题）
    if res is res_t:
        nu = res.params['nu']
        def tail_quantile(cl):
            return t.ppf(1 - cl, df=nu) * np.sqrt((nu - 2) / nu)   # 标准化 t 分位数
    else:
        def tail_quantile(cl):
            return norm.ppf(1 - cl)

    print("\n--- GARCH(1,1) 模型摘要（主分析区间，所选分布） ---")
    print(res.summary())

    alpha, beta = res.params['alpha[1]'], res.params['beta[1]']
    omega = res.params['omega']
    persistence = alpha + beta
    uncond_var_pct = omega / (1 - persistence)          # 无条件方差（百分比口径）
    uncond_vol_annual = np.sqrt(uncond_var_pct) / 100 * np.sqrt(252)
    half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
    print(f"\n波动率持续性 alpha+beta = {persistence:.3f}（冲击半衰期约 {half_life:.1f} 天）")
    print(f"模型隐含无条件年化波动率 = {uncond_vol_annual:.2%}（对照实际年化波动率 {ann_vol:.2%}）")

    cond_vol_pct = res.conditional_volatility
    cond_vol = cond_vol_pct / 100
    last_vol = cond_vol.iloc[-1]
    mu = port_returns.mean()
    dynamic_var_95 = mu + tail_quantile(0.95) * last_vol
    dynamic_var_99 = mu + tail_quantile(0.99) * last_vol

    static_var_95 = parametric_var_cvar(port_returns, 0.95)[0]
    static_var_99 = parametric_var_cvar(port_returns, 0.99)[0]

    print("\n--- 动态 VaR vs 静态 VaR（主分析区间） ---")
    print(f"最新 GARCH 条件波动率: {last_vol:.4f}")
    print(f"静态 95% VaR: {static_var_95:.4f}")
    print(f"动态 95% VaR: {dynamic_var_95:.4f}")
    print(f"静态 99% VaR: {static_var_99:.4f}")
    print(f"动态 99% VaR: {dynamic_var_99:.4f}")

    # ------------------------------
    # Kupiec POF 回测（样本内，动态 vs 静态）
    # ------------------------------
    print("\n--- Kupiec POF 回测（样本内） ---")
    n_obs = len(port_returns)
    for cl in (0.95, 0.99):
        z = tail_quantile(cl)                               # 与所选分布一致的分位数
        dyn_var_series = mu + z * cond_vol.values           # 逐日动态 VaR
        sta_var_series = mu + z * port_returns.std()        # 静态 VaR
        dyn_exc = int((port_returns.values < dyn_var_series).sum())
        sta_exc = int((port_returns.values < sta_var_series).sum())
        lr_d, p_d = kupiec_pof(dyn_exc, n_obs, cl)
        lr_s, p_s = kupiec_pof(sta_exc, n_obs, cl)
        print(f"\n置信水平 {cl:.0%}（理论突破率 {1-cl:.0%}）:")
        print(f"  静态 VaR: 突破 {sta_exc}/{n_obs} 次 ({sta_exc/n_obs:.2%})，Kupiec LR = {lr_s:.2f}, p = {p_s:.4f}")
        print(f"  动态 VaR: 突破 {dyn_exc}/{n_obs} 次 ({dyn_exc/n_obs:.2%})，Kupiec LR = {lr_d:.2f}, p = {p_d:.4f}")
    print("\n注：p > 0.05 表示突破频率与理论值无显著差异，模型通过回测。"
          "\n    此为样本内回测（模型见过全部数据），更严格的做法是滚动窗口样本外回测。")

    # ------------------------------
    # 压力测试：2020 年疫情（基于完整数据）
    # ------------------------------
    stress_start = '2020-02-20'
    stress_end = '2020-03-23'
    stress_period = port_returns_full[(port_returns_full.index >= stress_start) & (port_returns_full.index <= stress_end)]
    if len(stress_period) > 0:
        cum_loss = (1 + stress_period).prod() - 1          # 精确累计收益（路径连乘）
        worst_day = stress_period.min()                    # 单日最差
        stress_var_95 = np.percentile(stress_period, 5)
        stock_cummax = prices_full['stock'].cummax()
        stock_drawdown = (prices_full['stock'] - stock_cummax) / stock_cummax
        stress_max_drawdown = stock_drawdown.loc[stress_start:stress_end].min()

        hist_var_99, hist_cvar_99 = historical_var_cvar(port_returns, 0.99)

        print("\n--- 压力测试（2020 疫情） ---")
        print(f"压力期: {stress_start} 至 {stress_end}")
        print(f"压力期交易日数: {len(stress_period)}")
        print(f"压力期内组合累计收益率: {cum_loss:.2%}")
        print(f"压力期内组合平均日收益率: {stress_period.mean():.4f}")
        print(f"压力期内组合单日最差收益: {worst_day:.2%}")
        print(f"压力期内组合 5% 分位数: {stress_var_95:.4f}")
        print(f"股票部分最大回撤: {stress_max_drawdown:.2%}")
        print(f"对照：正常期 99% 历史 VaR = {hist_var_99:.4f}，"
              f"压力期单日最差收益{'超过' if worst_day < hist_var_99 else '未超过'}该阈值")
    else:
        print("\n压力测试时间段无数据，请检查完整数据范围。")

    # ------------------------------
    # 可视化（标题口径已修正为 2019-2026）
    # ------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1：主分析区间组合收益率分布与 VaR 阈值
    ax1 = axes[0, 0]
    ax1.hist(port_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=static_var_95, color='red', linestyle='--', label='95% VaR (static)')
    ax1.axvline(x=dynamic_var_95, color='darkred', linestyle='-', label='95% VaR (GARCH)')
    ax1.set_title('Portfolio Return Distribution and VaR (2019-2026)')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # 图2：主分析区间 GARCH 条件波动率
    ax2 = axes[0, 1]
    ax2.plot(cond_vol.index, cond_vol, color='blue', linewidth=0.8)
    ax2.set_title('GARCH(1,1) Conditional Volatility (2019-2026)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')
    ax2.grid(alpha=0.3)

    # 图3：蒙特卡洛模拟收益分布（正态 vs t）
    ax3 = axes[1, 0]
    np.random.seed(42)
    sim_norm = np.random.normal(mu, port_returns.std(), 10000)
    np.random.seed(42)
    z_t = np.random.standard_t(5, 10000) / np.sqrt(5 / 3)
    sim_t = mu + port_returns.std() * z_t
    ax3.hist(sim_norm, bins=50, alpha=0.5, color='lightgreen', edgecolor='black', label='Normal')
    ax3.hist(sim_t, bins=50, alpha=0.5, color='lightsalmon', edgecolor='black', label='Student-t (df=5)')
    ax3.axvline(x=np.percentile(sim_norm, 1), color='green', linestyle='--', label='99% VaR (normal)')
    ax3.axvline(x=np.percentile(sim_t, 1), color='red', linestyle='--', label='99% VaR (t)')
    ax3.set_title('Monte Carlo Simulated Returns: Normal vs Student-t (2019-2026)')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 图4：2020年疫情期组合累计收益（以压力窗口起点归一）
    ax4 = axes[1, 1]
    cum_ret_full = (1 + port_returns_full).cumprod()
    base = cum_ret_full.loc[:stress_start].iloc[-1]        # 压力期起点归一为 1
    stress_window = cum_ret_full.loc['2020-01-01':'2020-06-30'] / base
    if len(stress_window) > 0:
        ax4.plot(stress_window.index, stress_window, color='purple', label='Portfolio')
        stock_cum = (1 + returns_full['stock']).cumprod().loc['2020-01-01':'2020-06-30']
        stock_base = (1 + returns_full['stock']).cumprod().loc[:stress_start].iloc[-1]
        ax4.plot(stock_cum.index, stock_cum / stock_base, color='gray', alpha=0.7, label='Equity leg (CSI 300 ETF)')
        ax4.axhline(y=1.0, color='black', linewidth=0.5)
        ax4.set_title('Cumulative Return Since Stress-Window Start (2020 H1, base = 2020-02-20)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return (base = 1)')
        ax4.grid(alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No data for 2020 H1', ha='center', va='center')
        ax4.set_title('Stress Period Not Covered')

    plt.tight_layout()
    plt.savefig('portfolio_risk_analysis.png', dpi=150)
    plt.show()

    print("\n图表已保存为 portfolio_risk_analysis.png")
    print("项目完成！")
