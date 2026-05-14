# 信号与线性时不变系统

## 1. 信号的表示与分类

### 1.1 连续时间与离散时间信号

| 特征 | 连续时间信号 | 离散时间信号 |
|------|-------------|-------------|
| 表示 | $x(t)$, $t \in \mathbb{R}$ | $x[n]$, $n \in \mathbb{Z}$ |
| 采样关系 | — | $x[n] = x(nT_s)$ |
| 瞬时功率 | $p(t) = \|x(t)\|^2$ | $p[n] = \|x[n]\|^2$ |
| 总能量 | $E_\infty = \int_{-\infty}^{+\infty} \|x(t)\|^2 dt$ | $E_\infty = \sum_{n=-\infty}^{+\infty} \|x[n]\|^2$ |
| 平均功率 | $P_\infty = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{+T} \|x(t)\|^2 dt$ | $P_\infty = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} \|x[n]\|^2$ |

**信号分类**：
- **能量信号**：$0 < E_\infty < \infty$，$P_\infty = 0$
- **功率信号**：$0 < P_\infty < \infty$，$E_\infty = \infty$
- **周期信号**：$x(t+T) = x(t)$ 或 $x[n+N] = x[n]$
- **因果信号**：$t<0$ 或 $n<0$ 时为零

---

### 1.2 自变量变换

| 变换类型 | 连续时间 | 离散时间 |
|----------|---------|---------|
| 时移 | $x(t-t_0)$ | $x[n-n_0]$ |
| 反转 | $x(-t)$ | $x[-n]$ |
| 尺度变换 | $x(at)$ | $x[kn]$ ($k$ 为整数) |

**变换顺序**：先尺度变换和反转，最后时移。

**周期信号的特殊性**：
- 连续正弦 $x(t) = \cos(\omega_0 t + \phi)$ 总是周期的，$T = 2\pi/\omega_0$
- 离散正弦 $x[n] = \cos(\omega_0 n + \phi)$ 仅当 $\omega_0/2\pi$ 为有理数时周期

---

### 1.3 奇偶分解

任意信号可分解为偶部与奇部之和：

$$
x(t) = x_e(t) + x_o(t), \quad x_e(t) = \frac{1}{2}[x(t) + x(-t)], \quad x_o(t) = \frac{1}{2}[x(t) - x(-t)]
$$

**性质**：
- 奇信号在对称区间上积分为零
- 偶×偶=偶，奇×奇=偶，偶×奇=奇

---

## 2. 典型信号模型

### 2.1 复指数与正弦信号

| 信号类型 | 连续时间 | 离散时间 |
|----------|---------|---------|
| 复指数一般形式 | $x(t) = Ce^{at}$ | $x[n] = Cz^n$ |
| 复正弦 | $x(t) = Ce^{j\omega_0 t}$ | $x[n] = Ce^{j\omega_0 n}$ |
| 实正弦 | $x(t) = A\cos(\omega_0 t + \phi)$ | $x[n] = A\cos(\omega_0 n + \phi)$ |
| 频率范围 | $\omega_0 \in \mathbb{R}$ | $\omega_0 \in [-\pi, \pi]$ (周期 $2\pi$) |
| 周期性条件 | 总是周期 | $\omega_0/2\pi = m/N$ (有理数) |

**离散时间频率的特殊性**：
- $e^{j(\omega_0 + 2\pi)n} = e^{j\omega_0 n}$（频率周期性）
- $\omega_0 = 0$ 对应低频（直流），$\omega_0 = \pi$ 对应最高频率

---

### 2.2 单位冲激与单位阶跃

| 信号 | 连续时间 | 离散时间 |
|------|---------|---------|
| 单位阶跃 | $u(t) = \begin{cases} 1, & t > 0 \\ 0, & t < 0 \end{cases}$ | $u[n] = \begin{cases} 1, & n \geq 0 \\ 0, & n < 0 \end{cases}$ |
| 单位冲激 | $\delta(t) = 0 (t \neq 0), \int \delta(t) dt = 1$ | $\delta[n] = \begin{cases} 1, & n = 0 \\ 0, & n \neq 0 \end{cases}$ |
| 微分/差分关系 | $\delta(t) = \frac{du(t)}{dt}$ | $\delta[n] = u[n] - u[n-1]$ |
| 积分/求和关系 | $u(t) = \int_{-\infty}^{t} \delta(\tau) d\tau$ | $u[n] = \sum_{k=-\infty}^{n} \delta[k]$ |
| 筛选性质 | $\int x(t)\delta(t-t_0) dt = x(t_0)$ | $\sum x[k]\delta[n-k] = x[n]$ |
| 卷积性质 | $x(t) * \delta(t-t_0) = x(t-t_0)$ | $x[n] * \delta[n-n_0] = x[n-n_0]$ |

**连续时间冲激的尺度变换**：$\delta(at) = \frac{1}{|a|}\delta(t)$

---

## 3. 系统的基本性质

### 3.1 系统互联

| 互联方式 | 数学表示 |
|----------|---------|
| 级联 | $y(t) = H_2\{H_1\{x(t)\}\}$ |
| 并联 | $y(t) = H_1\{x(t)\} + H_2\{x(t)\}$ |
| 反馈 | $y(t) = H_1\{x(t) - H_2\{y(t)\}\}$ |

**性质**：级联顺序一般不可交换（LTI 系统除外），并联满足交换律和结合律。

---

### 3.2 系统性质的统一描述

| 性质 | 定义 | 连续/离散时间示例 |
|------|------|------------------|
| **无记忆** | 输出仅取决于当前输入 | $y(t) = x^2(t)$, $y[n] = 2x[n] + 3$ |
| **可逆** | 不同输入产生不同输出 | $y(t) = 2x(t)$ 可逆；$y(t) = x^2(t)$ 不可逆 |
| **因果** | 输出不依赖未来输入 | $y(t) = x(t) + x(t-1)$ 因果；$y(t) = x(t+1)$ 非因果 |
| **稳定 (BIBO)** | 有界输入产生有界输出 | $y(t) = e^{-t}x(t)$ 稳定；$y(t) = tx(t)$ 不稳定 |
| **时不变** | $x(t) \to y(t) \Rightarrow x(t-t_0) \to y(t-t_0)$ | $y(t) = 2x(t)$ 时不变；$y(t) = tx(t)$ 时变 |
| **线性** | $ax_1 + bx_2 \to ay_1 + by_2$ | $y(t) = 2x(t) + 3x(t-1)$ 线性；$y(t) = x^2(t)$ 非线性 |

**因果性注记**：所有实时物理系统必须是因果的。

**线性系统判断**：零输入必须产生零输出（$x(t) = 0 \Rightarrow y(t) = 0$）。

---

### 3.3 典型系统举例

| 系统类型 | 连续时间 | 离散时间 |
|----------|---------|---------|
| 恒等系统 | $y(t) = x(t)$ | $y[n] = x[n]$ |
| 延时系统 | $y(t) = x(t-t_0)$ | $y[n] = x[n-n_0]$ |
| 微分器/差分器 | $y(t) = \frac{dx(t)}{dt}$ | $y[n] = x[n] - x[n-1]$ |
| 积分器/累加器 | $y(t) = \int_{-\infty}^{t} x(\tau) d\tau$ | $y[n] = \sum_{k=-\infty}^{n} x[k]$ |
| 移动平均 | — | $y[n] = \frac{1}{M}\sum_{k=0}^{M-1} x[n-k]$ |
| RC 低通滤波器 | $RC\frac{dy(t)}{dt} + y(t) = x(t)$ | — |

---

## 4. 线性时不变系统：卷积表征

### 4.1 核心思想

LTI 系统同时满足**线性**和**时不变性**，其核心表征思想是：

> **任意输入可分解为基本信号的线性组合，系统输出等于这些基本信号响应的相同线性组合。**

| 系统类型 | 基本信号 | 分解方式 | 系统响应 |
|----------|---------|---------|---------|
| 离散时间 | 单位脉冲 $\delta[n]$ | 脉冲分解 | 卷积和 |
| 连续时间 | 单位冲激 $\delta(t)$ | 冲激分解 | 卷积积分 |

---

### 4.2 信号的基函数分解

**离散时间（脉冲分解）**：

$$
x[n] = \sum_{k=-\infty}^{+\infty} x[k]\delta[n-k]
$$

**连续时间（冲激分解）**：

$$
x(t) = \int_{-\infty}^{+\infty} x(\tau)\delta(t-\tau) d\tau
$$

---

### 4.3 卷积表示的统一推导

设系统对单位脉冲/冲激的响应为 $h[\cdot]$ 或 $h(\cdot)$。

| 步骤 | 离散时间 | 连续时间 |
|------|---------|---------|
| 输入分解 | $x[n] = \sum_k x[k]\delta[n-k]$ | $x(t) = \int x(\tau)\delta(t-\tau) d\tau$ |
| 时不变性 | $\delta[n-k] \to h[n-k]$ | $\delta(t-\tau) \to h(t-\tau)$ |
| 线性 | $x[k]\delta[n-k] \to x[k]h[n-k]$ | $x(\tau)\delta(t-\tau)d\tau \to x(\tau)h(t-\tau)d\tau$ |
| 叠加 | $\sum_k x[k]h[n-k]$ | $\int x(\tau)h(t-\tau) d\tau$ |

**统一结论**：

$$
\boxed{y[n] = x[n] * h[n] = \sum_{k=-\infty}^{+\infty} x[k]h[n-k]}
$$

$$
\boxed{y(t) = x(t) * h(t) = \int_{-\infty}^{+\infty} x(\tau)h(t-\tau) d\tau}
$$

---

### 4.4 卷积计算步骤对照

| 步骤 | 离散时间 | 连续时间 |
|------|---------|---------|
| 1 | 反转：$h[k] \to h[-k]$ | 反转：$h(\tau) \to h(-\tau)$ |
| 2 | 移位：$h[-k] \to h[n-k]$ | 移位：$h(-\tau) \to h(t-\tau)$ |
| 3 | 相乘：$x[k] \cdot h[n-k]$ | 相乘：$x(\tau) \cdot h(t-\tau)$ |
| 4 | 求和：$\sum_k$ | 积分：$\int d\tau$ |

**卷积长度性质**：若 $x[n]$ 长度为 $N$，$h[n]$ 长度为 $M$，则 $y[n]$ 长度为 $N+M-1$。

---

## 5. 卷积运算的代数性质

### 5.1 基本代数性质

| 性质 | 离散时间 | 连续时间 | 物理意义 |
|------|---------|---------|---------|
| **交换律** | $x[n] * h[n] = h[n] * x[n]$ | $x(t) * h(t) = h(t) * x(t)$ | 级联系统顺序可交换 |
| **分配律** | $x[n] * (h_1 + h_2) = x[n] * h_1 + x[n] * h_2$ | $x(t) * (h_1 + h_2) = x(t) * h_1 + x(t) * h_2$ | 并联系统等效为冲激响应相加 |
| **结合律** | $(x * h_1) * h_2 = x * (h_1 * h_2)$ | $(x * h_1) * h_2 = x * (h_1 * h_2)$ | 级联系统等效为冲激响应卷积 |

**级联系统等效**：

$$
x(t) \to \boxed{h_1(t)} \to \boxed{h_2(t)} \to y(t) \quad \Longleftrightarrow \quad x(t) \to \boxed{h_1(t) * h_2(t)} \to y(t)
$$

---

### 5.2 系统性质的冲激响应表征

| 系统性质 | 离散时间充要条件 | 连续时间充要条件 |
|----------|-----------------|-----------------|
| **无记忆** | $h[n] = K\delta[n]$ | $h(t) = K\delta(t)$ |
| **因果** | $h[n] = 0, \forall n < 0$ | $h(t) = 0, \forall t < 0$ |
| **稳定 (BIBO)** | $\sum_{k=-\infty}^{+\infty} \|h[k]\| < \infty$ | $\int_{-\infty}^{+\infty} \|h(\tau)\| d\tau < \infty$ |
| **可逆** | 存在 $h_i[n]$ 使 $h[n] * h_i[n] = \delta[n]$ | 存在 $h_i(t)$ 使 $h(t) * h_i(t) = \delta(t)$ |

**因果系统卷积简化**：

$$
y[n] = \sum_{k=0}^{+\infty} h[k]x[n-k], \quad y(t) = \int_{0}^{+\infty} h(\tau)x(t-\tau) d\tau
$$

**稳定性示例**：
- $h[n] = a^n u[n]$ 稳定 $\iff |a| < 1$
- $h(t) = e^{-t}u(t)$ 稳定（$\int_0^\infty e^{-t} dt = 1$）
- $h(t) = u(t)$ 不稳定（$\int_0^\infty 1 dt = \infty$）

---

### 5.3 阶跃响应

**单位阶跃响应** $s[\cdot]$ 或 $s(\cdot)$ 是系统对单位阶跃输入的响应：

| 关系 | 离散时间 | 连续时间 |
|------|---------|---------|
| 阶跃响应定义 | $s[n] = h[n] * u[n] = \sum_{k=-\infty}^{n} h[k]$ | $s(t) = h(t) * u(t) = \int_{-\infty}^{t} h(\tau) d\tau$ |
| 从阶跃响应恢复 | $h[n] = s[n] - s[n-1]$ | $h(t) = \frac{ds(t)}{dt}$ |

---

## 6. 微分方程与差分方程描述

### 6.1 线性常系数方程

| 方程类型 | 一般形式 |
|----------|---------|
| 微分方程（连续） | $\sum_{k=0}^{N} a_k \frac{d^k y(t)}{dt^k} = \sum_{k=0}^{M} b_k \frac{d^k x(t)}{dt^k}$ |
| 差分方程（离散） | $\sum_{k=0}^{N} a_k y[n-k] = \sum_{k=0}^{M} b_k x[n-k]$ |

**归一化形式**（$a_0 = 1$）：

$$
y[n] = -\sum_{k=1}^{N} a_k y[n-k] + \sum_{k=0}^{M} b_k x[n-k]
$$

**系统类型**：
- **递归（IIR）**：$N > 0$，输出依赖过去的输出
- **非递归（FIR）**：$N = 0$，输出仅依赖输入

**初始松弛条件**：因果 LTI 系统假设 $x(t) = 0$（或 $x[n] = 0$）对 $t < 0$（或 $n < 0$）时，$y(t) = 0$（或 $y[n] = 0$）。

---

### 6.2 示例：一阶系统

| 系统 | 方程 | 单位脉冲响应 | 稳定性条件 |
|------|------|-------------|-----------|
| 连续时间 RC 电路 | $\frac{dy(t)}{dt} + \frac{1}{RC}y(t) = \frac{1}{RC}x(t)$ | $h(t) = \frac{1}{RC}e^{-t/RC}u(t)$ | 总是稳定（$RC > 0$） |
| 离散时间一阶递归 | $y[n] - ay[n-1] = x[n]$ | $h[n] = a^n u[n]$ | $|a| < 1$ |

---

## 7. 奇异函数

### 7.1 单位冲激的数学本质

**离散时间**：$\delta[n]$ 是普通函数（单位脉冲序列）。

**连续时间**：$\delta(t)$ 是**广义函数**（分布），通过极限或卷积性质定义：

$$
\delta(t) = \lim_{\epsilon \to 0} \delta_\epsilon(t), \quad \int_{-\infty}^{+\infty} \delta_\epsilon(t) dt = 1
$$

**卷积定义**：$\delta(t)$ 满足 $x(t) * \delta(t) = x(t)$ 对任意"良好"函数成立。

---

### 7.2 冲激偶与高阶奇异函数

**单位冲激偶** $\delta'(t)$ 定义为 $\delta(t)$ 的一阶导数：

| 性质 | 公式 |
|------|------|
| 筛选性质 | $\int_{-\infty}^{+\infty} x(t)\delta'(t) dt = -x'(0)$ |
| 推广 | $\int_{-\infty}^{+\infty} x(t)\delta^{(k)}(t) dt = (-1)^k x^{(k)}(0)$ |
| 卷积性质 | $x(t) * \delta'(t) = x'(t)$ |
| 与阶跃关系 | $\delta'(t) = \frac{d^2 u(t)}{dt^2}$ |

**奇异函数族**（通过递归积分定义）：

$$
u_{-1}(t) = \delta(t), \quad u_0(t) = u(t), \quad u_k(t) = \frac{t^k}{k!}u(t) \ (k \geq 1)
$$

满足关系：

$$
\frac{d}{dt}u_k(t) = u_{k-1}(t), \quad u_k(t) = \int_{-\infty}^{t} u_{k-1}(\tau) d\tau
$$

---

## 8. 重要公式汇总

### 8.1 信号能量与功率

$$
\boxed{E_\infty = \int_{-\infty}^{+\infty} |x(t)|^2 dt = \sum_{n=-\infty}^{+\infty} |x[n]|^2}, \quad \boxed{P_\infty = \lim_{T \to \infty} \frac{1}{2T} \int_{-T}^{T} |x(t)|^2 dt}
$$

### 8.2 卷积运算

$$
\boxed{y[n] = x[n] * h[n] = \sum_{k=-\infty}^{+\infty} x[k]h[n-k]}, \quad \boxed{y(t) = x(t) * h(t) = \int_{-\infty}^{+\infty} x(\tau)h(t-\tau) d\tau}
$$

### 8.3 系统性质充要条件

$$
\boxed{\text{因果}: h[n]=0\ (\forall n<0) \iff h(t)=0\ (\forall t<0)}, \quad \boxed{\text{稳定}: \sum |h[k]|<\infty \iff \int |h(\tau)|d\tau<\infty}
$$

### 8.4 阶跃响应与冲激响应关系

$$
\boxed{h[n] = s[n] - s[n-1]}, \quad \boxed{h(t) = \frac{ds(t)}{dt}}
$$
