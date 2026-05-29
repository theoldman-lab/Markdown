# z变换

## 1. 引言：从离散时间傅里叶变换到 z 变换

### 1.1 DTFT 的局限与复指数序列的推广

DTFT $X(e^{j\omega}) = \sum_{n=-\infty}^{+\infty} x[n] e^{-j\omega n}$ 将序列分解为等幅复指数 $e^{j\omega n}$ 的线性组合。其收敛要求 $x[n]$ 绝对可和，导致 $u[n]$、$n u[n]$、$a^n u[n]\;(|a|>1)$ 等基本序列不存在 DTFT，且无法处理系统的初始状态。

> **核心矛盾**：求和核 $|e^{-j\omega n}| = 1$ 幅度恒为 1，对发散序列不加衰减。

引入复变量 $z = r e^{j\omega}$，考虑复指数序列 $z^n = r^n \cdot e^{j\omega n}$：$e^{j\omega n}$ 控制振荡频率，$r^n$ 控制增长（$r>1$）或衰减（$r<1$）。

**特征函数性质**：设离散 LTI 系统的单位样值响应为 $h[n]$，输入 $x[n] = z^n$ 时：
$$
y[n] = h[n] * z^n = z^n \sum_{k=-\infty}^{+\infty} h[k] z^{-k} = H(z) z^n
$$

即 $z^n$ 仍是离散 LTI 系统的特征函数，特征值 $H(z) = \sum h[k] z^{-k}$——这自然引出了 z 变换的定义。

---

### 1.2 基本思想与 DTFT 的关系

z 变换的核心：用衰减因子 $r^{-n}$ 乘序列，再做 DTFT。求和核从 $e^{-j\omega n}$ 推广为 $z^{-n} = r^{-n} \cdot e^{-j\omega n}$。

| 变换 | 求和核 | 路径 | 关系 |
|------|--------|------|------|
| DTFT | $e^{-j\omega n}$ | $z = e^{j\omega}$（单位圆） | $r = 1$ 的特例 |
| 双边 z 变换 | $z^{-n}$ | $z = r e^{j\omega}$ | 完整的 z 域分析 |
| 单边 z 变换 | $z^{-n}$ | $z = r e^{j\omega}$，$n \in [0, \infty)$ | 含初始条件的因果系统 |

三者关系：若 $X(z)$ 的 ROC 包含单位圆，则 DTFT $X(e^{j\omega}) = X(z)\big|_{z = e^{j\omega}}$；若 ROC 不包含单位圆（如增长序列 $a^n u[n], |a|>1$ 的 ROC 为 $|z|>|a|$），则 DTFT 不存在。

$$
\boxed{\mathcal{F}\{x[n]\} = X(e^{j\omega}) = X(z)\big|_{z = e^{j\omega}}, \quad \text{当 } |z| = 1 \in \text{ROC}}
$$

> **z 变换是 DTFT 的推广，DTFT 是 z 变换在单位圆上的限制。**

---

### 1.3 s 平面到 z 平面的映射

对连续信号 $x(t)$ 以采样间隔 $T$ 理想采样，$x[n] = x(nT)$ 的 z 变换与拉普拉斯变换的关系为 $z = e^{sT}$：

$$
z = e^{sT} = e^{(\sigma + j\omega)T} = e^{\sigma T} \cdot e^{j\omega T}
$$

| s 平面区域 | 映射条件 | z 平面区域 |
|------------|----------|-----------|
| 虚轴 $\Re\{s\} = 0$ | $|z| = 1$ | **单位圆** |
| 左半平面 $\Re\{s\} < 0$ | $|z| < 1$ | **单位圆内部** |
| 右半平面 $\Re\{s\} > 0$ | $|z| > 1$ | **单位圆外部** |

**关键结论**：
- s 左半平面（稳定）→ z 单位圆内部（稳定），虚轴 → 单位圆
- $s \to s + j 2\pi/T$ 映射到同一个 $z$，体现采样带来的频谱周期性

---

## 2. z 变换的定义

### 2.1 双边 z 变换

离散时间序列 $x[n]$ 的**双边 z 变换**（Bilateral z-Transform）定义为：

$$
\boxed{X(z) = \mathcal{Z}\{x[n]\} = \sum_{n=-\infty}^{+\infty} x[n] z^{-n}}
$$

其中 $z = r e^{j\omega}$ 为复变量，求和在对使该级数收敛的 z 值上进行。

**关键要点**：

- 变换结果 $X(z)$ 是复变量 $z$ 的复变函数
- 求和从 $n = -\infty$ 到 $n = +\infty$，涵盖序列的全部历史
- $X(z)$ 只在使级数**绝对收敛**的 $z$ 范围内有意义，这一范围称为**收敛域**（ROC）

**记号约定**：$x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$ 表示 z 变换对。

---

### 2.2 单边 z 变换

**单边 z 变换**（Unilateral z-Transform）定义为：

$$
\boxed{\mathcal{X}(z) = \mathcal{Z}_u\{x[n]\} = \sum_{n=0}^{\infty} x[n] z^{-n}}
$$

**与双边变换的区别**：

| 特性 | 双边 z 变换 | 单边 z 变换 |
|------|------------|-------------|
| 求和区间 | $(-\infty, +\infty)$ | $[0, +\infty)$ |
| 对 $n < 0$ 的序列值 | 纳入变换 | **忽略** |
| 初始条件处理 | 不便处理 | **可自然纳入** |
| 唯一性问题 | $x[n]$ 与 $X(z)$ + ROC 一一对应 | $x[n]$（$n \geq 0$ 部分）与 $\mathcal{X}(z)$ 一一对应 |
| 主要应用 | 序列与系统的完整理论分析 | 含初始条件的差分方程求解 |

> **注意**：若 $x[n]$ 满足 $x[n] = 0$ 对 $n < 0$（因果序列），则双边变换与单边变换一致（在共同的 ROC 内）。此时 $X(z) = \mathcal{X}(z)$。

---

### 2.3 z 平面的几何意义

复变量 $z = r e^{j\omega}$ 在复平面上形成一个二维坐标系，称为 **z 平面**（z-plane）：

| 坐标 | 记号 | 物理意义 |
|------|------|----------|
| 模 $r = |z|$ | 序列的衰减/增长因子，$r > 1$ 对应增长序列，$r < 1$ 对应衰减序列 |
| 辐角 $\omega = \angle z$ | 归一化频率 $[0, 2\pi)$ | 振荡频率，$\omega$ 越大振荡越快 |

**z 平面上的关键区域**：

- **单位圆**（$|z| = 1$）：$z = e^{j\omega}$，对应 DTFT 的求和路径，幅频特性直接在此圆上求值
- **单位圆内部**（$|z| < 1$）：对应衰减的复指数序列，求和更易收敛
- **单位圆外部**（$|z| > 1$）：对应增长的复指数序列，求和更难收敛
- **原点**（$z = 0$）：只影响序列的超前/延迟，不影响收敛性
- **无穷远点**（$z = \infty$）：某些序列在该处有极点或零点

**收敛域在 z 平面上的表示**：

与拉普拉斯变换不同，z 变换的 ROC 由**以原点为圆心的圆环区域**构成——级数的收敛仅取决于 $|z|$（模），而不取决于辐角 $\omega$。

---

## 3. 收敛域（ROC）

### 3.1 收敛域的定义与基本性质

**收敛域**（Region of Convergence, ROC）定义为使 z 变换级数绝对收敛的所有复变量 $z$ 的集合：

$$
\boxed{\text{ROC} = \left\{ z \in \mathbb{C} \;\bigg|\; \sum_{n=-\infty}^{+\infty} |x[n] z^{-n}| = \sum_{n=-\infty}^{+\infty} |x[n]| \, |z|^{-n} < \infty \right\}}
$$

**关键观察**：

$$
\left| x[n] z^{-n} \right| = \left| x[n] \right| \cdot |z|^{-n} = |x[n]| \, r^{-n}
$$

收敛条件仅与 $z$ 的模 $r = |z|$ 有关，辐角 $\omega = \angle z$ 不参与绝对值判断。因此：

> **ROC 在 z 平面上由以原点为圆心的圆环区域构成。**

以下是 ROC 的核心性质：

**性质 1**：ROC 由 z 平面上以原点为圆心的圆环组成。

$$
\text{ROC} = \{ z \in \mathbb{C} \mid r_L < |z| < r_H \}, \quad 0 \leq r_L < r_H \leq \infty
$$

**性质 2**：对于有理 z 变换，ROC 内不包含任何极点（$X(z)$ 在极点处趋于无穷大，不可能满足绝对可和条件）。

**性质 3**：若 $x[n]$ 为有限长序列（在有限区间外为零），且 $X(z)$ 至少在某个 $z$ 值处绝对可和，则 ROC 为整个 z 平面，可能除去 $z = 0$ 和/或 $z = \infty$。

具体而言：
- 若 $x[n]$ 仅在 $n \geq 0$ 范围非零（因果有限长），ROC 为 $|z| > 0$
- 若 $x[n]$ 仅在 $n \leq 0$ 范围非零（反因果有限长），ROC 为 $|z| < \infty$
- 若 $x[n]$ 在 $n < 0$ 和 $n > 0$ 均有非零值，ROC 为 $0 < |z| < \infty$

**性质 4**：若 $X(z)$ 是有理函数，则 ROC 的边界由以原点为圆心的圆周（$|z| = \text{极点模}$）确定，ROC 不包含边界。

---

### 3.2 不同序列类型的收敛域

序列的"非零区间形状"决定了 ROC 的几何形态。

#### 3.2.1 右边序列（含因果序列）

**定义**：$x[n]$ 为**右边序列**（Right-sided Sequence），如果存在 $N_1$ 使得 $n < N_1$ 时 $x[n] = 0$。最常见的特例是因果序列（$N_1 = 0$）。

**ROC 特征**：

> 若 $z_0 \in \text{ROC}$，且 $|z| > |z_0|$，则 $z$ 也在 ROC 内。

**结论**：右边序列的 ROC 是某个圆的外部区域（可能含无穷远点）：

$$
\boxed{\text{ROC}: |z| > r_{\min}}
$$

其中 $r_{\min}$ 为 $X(z)$ 的最大极点的模。

**证明思路**：设 $z_0 \in \text{ROC}$，$r = |z| > |z_0| = r_0$。对因果序列（$N_1 = 0$）：

$$
\begin{aligned}
\sum_{n=-\infty}^{+\infty} |x[n]| r^{-n}
&= \sum_{n=0}^{\infty} |x[n]| r_0^{-n} \cdot \left(\frac{r_0}{r}\right)^n \\
&\leq \sum_{n=0}^{\infty} |x[n]| r_0^{-n} < \infty
\end{aligned}
$$

由于 $r_0/r < 1$，附加的因子 $(r_0/r)^n$ 使级数更小，收敛性保持。

---

#### 3.2.2 左边序列（含反因果序列）

**定义**：$x[n]$ 为**左边序列**（Left-sided Sequence），如果存在 $N_2$ 使得 $n > N_2$ 时 $x[n] = 0$。

**ROC 特征**：

> 若 $z_0 \in \text{ROC}$，且 $|z| < |z_0|$，则 $z$ 也在 ROC 内。

**结论**：左边序列的 ROC 是某个圆的内部区域（可能含原点）：

$$
\boxed{\text{ROC}: |z| < r_{\max}}
$$

其中 $r_{\max}$ 为 $X(z)$ 的最小极点的模。

**证明思路**：与右边序列对称。设 $N_2 = 0$（反因果序列：$n > 0$ 时 $x[n] = 0$），级数范围为 $n \leq 0$。当 $r < r_0$ 时，$(r_0/r)^n \leq 1$（因 $n \leq 0$ 且 $r_0/r > 1$），收敛性保持。

---

#### 3.2.3 双边序列

**定义**：$x[n]$ 在 $n \to -\infty$ 和 $n \to +\infty$ 方向均无限延伸。

**分析**：将 $x[n]$ 分解为右边序列与左边序列之和：

$$
x[n] = x_R[n] + x_L[n]
$$

由线性性质：

$$
X(z) = X_R(z) + X_L(z), \quad \text{ROC} = \text{ROC}_R \cap \text{ROC}_L
$$

**结论**：双边序列的 ROC 是**圆环区域**：

$$
\boxed{\text{ROC} = \{ z \in \mathbb{C} \mid r_R < |z| < r_L \}}
$$

其中 $r_R$ 由 $X_R(z)$ 的最大极点模决定，$r_L$ 由 $X_L(z)$ 的最小极点模决定。

> **当且仅当 $r_R < r_L$ 时，ROC 非空，z 变换存在。**

**典型例子**：$x[n] = a^{|n|}$（$|a| < 1$）。右边部 $a^n u[n]$ 的 ROC 为 $|z| > |a|$，左边部 $a^{-n} u[-n-1]$ 的 ROC 为 $|z| < 1/|a|$。当 $|a| < 1$ 时 $|a| < 1/|a|$，交集非空。

---

#### 3.2.4 有限长序列

**定义**：$x[n]$ 在某个有限区间 $[N_1, N_2]$ 之外恒为零。

**ROC 特征**：对于有限长序列，z 变换为有限项求和，每一项 $x[n] z^{-n}$ 在任意有限 $z$ 处均绝对可和。

> **有限长序列的 ROC 为整个 z 平面（可能除去 $z = 0$ 和/或 $z = \infty$）。**

| 有限长序列的非零范围 | ROC |
|----------------------|-----|
| 仅 $n \geq 0$（因果有限长） | $|z| > 0$（除去 $z = 0$） |
| 仅 $n \leq 0$（反因果有限长） | $|z| < \infty$（除去 $\infty$） |
| 跨越 $n = 0$ 两侧 | $0 < |z| < \infty$（除去 $0$ 和 $\infty$） |

---

### 3.3 ROC 与序列唯一性的关系

同一 $X(z)$ 的代数表达式，配以不同的 ROC，对应完全不同的时域序列。**z 变换的唯一性要求同时给定 $X(z)$ 和 ROC**。

**典型例子**：考虑 $X(z) = \dfrac{1}{1 - 2z^{-1}} = \dfrac{z}{z - 2}$（极点在 $z = 2$）。两种可能的 ROC 对应两种不同的序列：

| ROC | 序列类型 | 时域表达式 |
|-----|----------|-----------|
| $|z| > 2$ | 右边（因果）序列 | $x[n] = 2^n u[n]$ |
| $|z| < 2$ | 左边（反因果）序列 | $x[n] = -2^n u[-n-1]$ |

---

### 3.4 零极点与 ROC

**定义**：

- **零点**（Zero）：使 $X(z) = 0$ 的 $z$ 值
- **极点**（Pole）：使 $X(z) \to \infty$ 的 $z$ 值

对于有理 z 变换：

$$
X(z) = \frac{N(z)}{D(z)} = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}
$$

- 零点 = 分子多项式 $N(z)$ 的根
- 极点 = 分母多项式 $D(z)$ 的根

**收敛域的边界由极点确定**：

| 序列类型 | ROC 形状 | 边界由谁确定 |
|----------|----------|-------------|
| 右边序列 | $|z| > r_{\min}$ | **最大极点模**（最外极点的模） |
| 左边序列 | $|z| < r_{\max}$ | **最小极点模**（最内极点的模） |
| 双边序列 | $r_R < |z| < r_L$ | 分别由右边部的最大极点模（$r_R$）和左边部的最小极点模（$r_L$）确定 |

**因果性与稳定性的初步判据**（详见第 8 章）：

- 若系统是**因果的**，则 $h[n]$ 是右边序列，ROC 为圆外区域（$|z| > r_{\min}$）
- 若系统是**稳定的**（$h[n]$ 绝对可和），则 DTFT 存在，ROC 必须包含单位圆

> **因果稳定系统的 ROC 包含单位圆，且为 $|z| > r_{\min}$ 的形式。这意味着因果稳定系统的所有极点必须位于 z 平面单位圆内部（$r_{\min} < 1$）。**

---

## 4. 常见序列的 z 变换

### 4.1 单位样值序列 $\delta[n]$

单位样值序列 $\delta[n]$ 是最基本的离散序列，其 z 变换通过筛选性质直接得出：

$$
\begin{aligned}
\mathcal{Z}\{\delta[n]\} &= \sum_{n=-\infty}^{+\infty} \delta[n] z^{-n} = z^{0} = 1
\end{aligned}
$$

$$
\boxed{\delta[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} 1, \quad \text{ROC}: \text{整个 z 平面}}
$$

> 单位样值序列的 z 变换在所有 z 处恒为 1——它均匀包含所有复频率分量。

**时移样值**：$\delta[n - n_0]$ 的 z 变换为 $z^{-n_0}$（$n_0 > 0$ 时 ROC 除去 $z = 0$，$n_0 < 0$ 时 ROC 除去 $z = \infty$）。

---

### 4.2 单位阶跃序列 $u[n]$

单位阶跃 $u[n]$ 是因果系统中最重要的基本序列。其 z 变换：

$$
\begin{aligned}
\mathcal{Z}\{u[n]\} &= \sum_{n=0}^{\infty} 1 \cdot z^{-n} = \sum_{n=0}^{\infty} (z^{-1})^n
\end{aligned}
$$

这是一个几何级数，求和收敛的充要条件是 $|z^{-1}| < 1$，即 $|z| > 1$：

$$
\mathcal{Z}\{u[n]\} = \frac{1}{1 - z^{-1}} = \frac{z}{z - 1}, \quad |z| > 1
$$

$$
\boxed{u[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{1}{1 - z^{-1}} = \frac{z}{z - 1}, \quad \text{ROC}: |z| > 1}
$$

> $u[n]$ 的 DTFT 不存在（不满足绝对可和），但其 z 变换存在（$|z| > 1$ 提供了衰减因子）。极点在 $z = 1$（单位圆上）。

---

### 4.3 单边指数序列 $a^n u[n]$

单边指数序列是 z 变换中最具代表性的序列。推导如下：

$$
\begin{aligned}
\mathcal{Z}\{a^n u[n]\} &= \sum_{n=0}^{\infty} a^n z^{-n} = \sum_{n=0}^{\infty} (a z^{-1})^n
\end{aligned}
$$

几何级数收敛的充要条件为 $|a z^{-1}| < 1$，即 $|z| > |a|$：

$$
\mathcal{Z}\{a^n u[n]\} = \frac{1}{1 - a z^{-1}} = \frac{z}{z - a}, \quad |z| > |a|
$$

$$
\boxed{a^n u[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{1}{1 - a z^{-1}} = \frac{z}{z - a}, \quad \text{ROC}: |z| > |a|}
$$

**特殊情况分析**：

| 参数 $a$ | 时域序列 | $X(z)$ | 极点位置 | ROC |
|----------|----------|--------|----------|-----|
| $|a| < 1$ | 衰减指数 | $\dfrac{z}{z - a}$ | $z = a$（单位圆内） | $|z| > |a|$（含单位圆） |
| $a = 1$ | 单位阶跃 | $\dfrac{z}{z - 1}$ | $z = 1$（单位圆上） | $|z| > 1$ |
| $|a| > 1$ | 增长指数 | $\dfrac{z}{z - a}$ | $z = a$（单位圆外） | $|z| > |a|$（不含单位圆） |

> **DTFT 的存在性**：当 $|a| < 1$ 时 ROC 包含单位圆，DTFT 存在且为 $\dfrac{1}{1 - a e^{-j\omega}}$；当 $|a| \geq 1$ 时，DTFT 不存在。

---

### 4.4 正弦与余弦序列

利用欧拉公式将正弦/余弦表示为复指数序列的线性组合，再借助 $a^n u[n]$ 的结果。

**余弦序列 $\cos(\omega_0 n) u[n]$**：

$$
\cos(\omega_0 n) u[n] = \frac{1}{2} \big( e^{j\omega_0 n} + e^{-j\omega_0 n} \big) u[n]
$$

由线性性质与 $a^n u[n]$ 的结果（$a = e^{\pm j\omega_0}$）：

$$
\begin{aligned}
\mathcal{Z}\{\cos(\omega_0 n) u[n]\} &= \frac{1}{2} \left( \frac{1}{1 - e^{j\omega_0} z^{-1}} + \frac{1}{1 - e^{-j\omega_0} z^{-1}} \right) \\
&= \frac{1 - \cos(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}, \quad |z| > 1
\end{aligned}
$$

**正弦序列 $\sin(\omega_0 n) u[n]$**：

$$
\sin(\omega_0 n) u[n] = \frac{1}{2j} \big( e^{j\omega_0 n} - e^{-j\omega_0 n} \big) u[n]
$$

同理：

$$
\mathcal{Z}\{\sin(\omega_0 n) u[n]\} = \frac{\sin(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}, \quad |z| > 1
$$

$$
\boxed{
\begin{aligned}
\cos(\omega_0 n) u[n] &\stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{1 - \cos(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}, \quad |z| > 1 \\[6pt]
\sin(\omega_0 n) u[n] &\stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{\sin(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}, \quad |z| > 1
\end{aligned}
}
$$

**零极点分析**：两种序列的极点均为 $z = e^{\pm j\omega_0}$（在单位圆上），因此 ROC 为 $|z| > 1$。这与连续时间正弦信号的拉普拉斯变换极点位于虚轴对应。

---

### 4.5 斜坡序列 $n^k u[n]$

**一阶斜坡 $n u[n]$**：利用 z 域微分性质（见第 5 章）：

$$
\mathcal{Z}\{n u[n]\} = -z \frac{d}{dz} \left( \frac{1}{1 - z^{-1}} \right) = \frac{z^{-1}}{(1 - z^{-1})^2} = \frac{z}{(z - 1)^2}, \quad |z| > 1
$$

$$
\boxed{n u[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{z^{-1}}{(1 - z^{-1})^2} = \frac{z}{(z - 1)^2}, \quad |z| > 1}
$$

> $n^k u[n]$ 在 $z = 1$ 处有高阶极点。尽管序列随时间增长无界，其 z 变换依然存在（$|z| > 1$）。

---

### 4.6 常见变换对速查表

| 编号 | 时域序列 $x[n]$ | z 域变换 $X(z)$ | ROC |
|------|----------------|----------------|-----|
| 1 | $\delta[n]$ | $1$ | 整个 z 平面 |
| 2 | $\delta[n - n_0]$ | $z^{-n_0}$ | $n_0>0$：除 $z=0$；$n_0<0$：除 $\infty$ |
| 3 | $u[n]$ | $\dfrac{1}{1 - z^{-1}} = \dfrac{z}{z - 1}$ | $|z| > 1$ |
| 4 | $-u[-n-1]$ | $\dfrac{1}{1 - z^{-1}} = \dfrac{z}{z - 1}$ | $|z| < 1$ |
| 5 | $a^n u[n]$ | $\dfrac{1}{1 - a z^{-1}} = \dfrac{z}{z - a}$ | $|z| > |a|$ |
| 6 | $-a^n u[-n-1]$ | $\dfrac{1}{1 - a z^{-1}} = \dfrac{z}{z - a}$ | $|z| < |a|$ |
| 7 | $n a^n u[n]$ | $\dfrac{a z^{-1}}{(1 - a z^{-1})^2} = \dfrac{az}{(z - a)^2}$ | $|z| > |a|$ |
| 8 | $n u[n]$ | $\dfrac{z^{-1}}{(1 - z^{-1})^2} = \dfrac{z}{(z - 1)^2}$ | $|z| > 1$ |
| 9 | $\cos(\omega_0 n) u[n]$ | $\dfrac{1 - \cos(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}$ | $|z| > 1$ |
| 10 | $\sin(\omega_0 n) u[n]$ | $\dfrac{\sin(\omega_0) z^{-1}}{1 - 2\cos(\omega_0) z^{-1} + z^{-2}}$ | $|z| > 1$ |
| 11 | $r^n \cos(\omega_0 n) u[n]$ | $\dfrac{1 - r\cos(\omega_0) z^{-1}}{1 - 2r\cos(\omega_0) z^{-1} + r^2 z^{-2}}$ | $|z| > |r|$ |
| 12 | $r^n \sin(\omega_0 n) u[n]$ | $\dfrac{r\sin(\omega_0) z^{-1}}{1 - 2r\cos(\omega_0) z^{-1} + r^2 z^{-2}}$ | $|z| > |r|$ |

> **注意**：第 4 行和第 6 行是对应左边（反因果）序列的变换，其 ROC 与因果版本对称。这再次说明 ROC 在唯一确定序列中的关键作用。

---

## 5. z 变换的性质

### 5.1 线性

若 $x_1[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X_1(z)$，ROC = $R_1$；$x_2[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X_2(z)$，ROC = $R_2$，则：

$$
\boxed{a x_1[n] + b x_2[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} a X_1(z) + b X_2(z), \quad \text{ROC} \supseteq R_1 \cap R_2}
$$

ROC 可能因零极点相消而扩大。

---

### 5.2 时移性质

时移性质是 z 变换区别于拉普拉斯变换最具特色的性质之一——它自然地将离散系统中的"延迟"操作表示为乘以 $z^{-1}$。

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则：

$$
\boxed{x[n - n_0] \stackrel{\mathcal{Z}}{\longleftrightarrow} z^{-n_0} X(z), \quad \text{ROC} = R \text{（可能增减 } z = 0 \text{ 或 } z = \infty\text{）}}
$$

**证明（双边变换）**：

$$
\begin{aligned}
\mathcal{Z}\{x[n - n_0]\} &= \sum_{n=-\infty}^{+\infty} x[n - n_0] z^{-n} \\
&= \sum_{m=-\infty}^{+\infty} x[m] z^{-(m + n_0)} \quad (m = n - n_0) \\
&= z^{-n_0} \sum_{m=-\infty}^{+\infty} x[m] z^{-m} = z^{-n_0} X(z)
\end{aligned}
$$

**物理意义**：
- $n_0 > 0$（延迟）：乘以 $z^{-n_0}$——引入 $n_0$ 个样本的延迟
- $n_0 < 0$（超前）：乘以 $z^{-n_0}$（$n_0 < 0$ 时幂次为正）——在因果实现中不可物理实现

> **对 ROC 的影响**：$z^{-n_0}$ 可能引入 $z = 0$（$n_0 > 0$）或 $z = \infty$（$n_0 < 0$）处的极点/零点。单边 z 变换的时移表达式与序列支撑范围有关，见第 7 章。

---

### 5.3 z 域尺度变换（序列乘以指数）

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则：

$$
\boxed{z_0^n \, x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X\!\left(\frac{z}{z_0}\right), \quad \text{ROC} = |z_0| \cdot R}
$$

**证明**：

$$
\begin{aligned}
\mathcal{Z}\{z_0^n \, x[n]\} &= \sum_{n=-\infty}^{+\infty} z_0^n x[n] z^{-n} \\
&= \sum_{n=-\infty}^{+\infty} x[n] \left( \frac{z}{z_0} \right)^{-n} = X\!\left( \frac{z}{z_0} \right)
\end{aligned}
$$

---

### 5.4 时间反转

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则：

$$
\boxed{x[-n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z^{-1}), \quad \text{ROC} = \frac{1}{R}}
$$

其中 $1/R$ 表示将 ROC 中每点的模取倒数后的区域。

**证明**：

$$
\mathcal{Z}\{x[-n]\} = \sum_{n=-\infty}^{+\infty} x[-n] z^{-n} = \sum_{m=-\infty}^{+\infty} x[m] (z^{-1})^{-m} = X(z^{-1})
$$

**物理意义**：时域反转映射为 z 域中 $z \to z^{-1}$ 的变换。若原序列的 ROC 为 $r_L < |z| < r_H$，则反转序列的 ROC 为 $1/r_H < |z| < 1/r_L$。

---

### 5.5 z 域微分

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则：

$$
\boxed{n x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} -z \frac{dX(z)}{dz}, \quad \text{ROC} = R}
$$

**证明**：

$$
\frac{dX(z)}{dz} = \frac{d}{dz} \sum_{n=-\infty}^{+\infty} x[n] z^{-n} = \sum_{n=-\infty}^{+\infty} x[n] (-n) z^{-n-1} = -z^{-1} \sum_{n=-\infty}^{+\infty} n x[n] z^{-n}
$$

两边乘以 $-z$ 即得证（在 ROC 内级数一致收敛，求导与求和可交换）。

**推广**：

$$
\boxed{n^k x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} \left( -z \frac{d}{dz} \right)^k X(z)}
$$

---

### 5.6 共轭对称性

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则对于复共轭序列：

$$
\boxed{x^*[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X^*(z^*), \quad \text{ROC} = R}
$$

**推论**：实序列的 z 变换满足 $X(z) = X^*(z^*)$。这意味着若 $z_0$ 为极点/零点，则 $z_0^*$ 也是极点/零点，即复零极点必以共轭对形式出现。

---

### 5.7 卷积定理

卷积定理是 z 变换作为离散 LTI 系统分析工具最核心的性质。

若 $x_1[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X_1(z)$，ROC = $R_1$；$x_2[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X_2(z)$，ROC = $R_2$，则：

$$
\boxed{x_1[n] * x_2[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X_1(z) X_2(z), \quad \text{ROC} \supseteq R_1 \cap R_2}
$$

**证明**：

$$
\begin{aligned}
\mathcal{Z}\{x_1[n] * x_2[n]\} &= \sum_{n=-\infty}^{+\infty} \left[ \sum_{k=-\infty}^{+\infty} x_1[k] x_2[n - k] \right] z^{-n} \\
&= \sum_{k=-\infty}^{+\infty} x_1[k] \sum_{n=-\infty}^{+\infty} x_2[n - k] z^{-n} \\
&= \sum_{k=-\infty}^{+\infty} x_1[k] z^{-k} X_2(z) \quad (\text{时移性质}) \\
&= X_1(z) X_2(z)
\end{aligned}
$$

> **时域卷积 ⇔ z 域相乘**。这一性质将离散 LTI 系统的输出计算从卷积和转化为代数乘法（详见第 8 章）。

---

### 5.8 序列累加

若 $x[n] \stackrel{\mathcal{Z}}{\longleftrightarrow} X(z)$，ROC = $R$，则：

$$
\boxed{\sum_{k=-\infty}^{n} x[k] \stackrel{\mathcal{Z}}{\longleftrightarrow} \frac{1}{1 - z^{-1}} X(z), \quad \text{ROC} \supseteq R \cap \{|z| > 1\}}
$$

**证明**：累加可写为卷积形式 $\sum_{k=-\infty}^{n} x[k] = x[n] * u[n]$，由卷积定理和 $u[n] \longleftrightarrow 1/(1 - z^{-1})$（$|z| > 1$）直接得出。

> **直观解释**：时域累加对应 z 域乘以 $1/(1 - z^{-1})$——累加等价于与阶跃序列卷积，阶跃序列充当离散积分器。

---

### 5.9 初值定理与终值定理

这两条定理允许直接从 z 域表达式获取因果序列在 $n = 0$ 和 $n \to \infty$ 处的行为，无需完整反变换。

**初值定理**（Initial Value Theorem）：

若 $x[n]$ 是因果序列（$n < 0$ 时 $x[n] = 0$），则：

$$
\boxed{x[0] = \lim_{z \to \infty} X(z)}
$$

**证明**：对于因果序列，$X(z) = x[0] + x[1] z^{-1} + x[2] z^{-2} + \cdots$。当 $z \to \infty$ 时，$z^{-k} \to 0$（$k \geq 1$），因此 $\lim_{z \to \infty} X(z) = x[0]$。

**终值定理**（Final Value Theorem）：

若 $x[n]$ 是因果序列且 $\lim_{n \to \infty} x[n] < \infty$ 存在，且 $(z - 1)X(z)$ 的极点都在单位圆内，则：

$$
\boxed{\lim_{n \to \infty} x[n] = \lim_{z \to 1} (1 - z^{-1}) X(z) = \lim_{z \to 1} (z - 1) X(z)}
$$



> **终值定理的有效性条件**：$(z - 1)X(z)$ 在单位圆上及外部（除 $z = 1$ 可能为单极点外）无极点。若不满足（如 $X(z)$ 的极点在单位圆上），结论不可靠——此时 $x[n]$ 为持续振荡，无稳态终值。

---

### 5.10 性质总结表

| 编号 | 性质 | 时域 | z 域 | ROC |
|------|------|------|------|-----|
| 1 | 线性 | $a x_1[n] + b x_2[n]$ | $a X_1(z) + b X_2(z)$ | $\supseteq R_1 \cap R_2$ |
| 2 | 时移 | $x[n - n_0]$ | $z^{-n_0} X(z)$ | $R$（可能增减 $0$ 或 $\infty$） |
| 3 | z 域尺度变换 | $z_0^n x[n]$ | $X(z/z_0)$ | $|z_0| \cdot R$ |
| 4 | 时间反转 | $x[-n]$ | $X(z^{-1})$ | $1/R$ |
| 5 | z 域微分 | $n x[n]$ | $-z \dfrac{dX(z)}{dz}$ | $R$ |
| 6 | 共轭 | $x^*[n]$ | $X^*(z^*)$ | $R$ |
| 7 | 卷积 | $x_1[n] * x_2[n]$ | $X_1(z) X_2(z)$ | $\supseteq R_1 \cap R_2$ |
| 8 | 序列累加 | $\displaystyle\sum_{k=-\infty}^{n} x[k]$ | $\dfrac{1}{1 - z^{-1}} X(z)$ | $\supseteq R \cap \{|z| > 1\}$ |
| 9 | 初值定理 | $x[0]$ | $\displaystyle\lim_{z \to \infty} X(z)$ | —（因果序列） |
| 10 | 终值定理 | $\displaystyle\lim_{n \to \infty} x[n]$ | $\displaystyle\lim_{z \to 1} (1 - z^{-1}) X(z)$ | —（极点条件见 5.9 节） |
| 11 | 时移（单边） | $x[n-1]$ | $z^{-1}\mathcal{X}(z) + x[-1]$ | — |

---

## 6. z 反变换

### 6.1 反演积分

z 反变换将 z 域函数 $X(z)$ 恢复为时域序列 $x[n]$。其形式定义由围道积分给出：

$$
\boxed{x[n] = \mathcal{Z}^{-1}\{X(z)\} = \frac{1}{2\pi j} \oint_{C} X(z) z^{n-1} dz}
$$

其中 $C$ 是 z 平面上一条包围原点、且完全位于 ROC 内的逆时针闭合围道。

工程实践中，几乎总是通过以下三种等价方法完成反变换：

1. **部分分式展开 + 查表**（适用于有理 $X(z)$，6.2 节）
2. **幂级数展开 / 长除法**（适用于求前几项，6.3 节）
3. **留数法**（一般方法，6.4 节）

---

### 6.2 部分分式展开法

部分分式展开是求有理 z 变换反变换最常用的方法。有两种等价的展开策略：

- **策略一**：以 $z^{-1}$ 为变量展开（直接匹配 $1/(1 - a z^{-1})$ 形式）
- **策略二**：先对 $X(z)/z$ 展开，再乘以 $z$（传统方法，便于复用 s 域部分分式经验）

以下以策略二为主进行讨论。

#### 6.2.1 单极点

设 $X(z)$ 有 $N$ 个互不相同的单极点 $p_1, p_2, \ldots, p_N$。先求 $X(z)/z$ 的部分分式展开：

$$
\frac{X(z)}{z} = \sum_{k=1}^{N} \frac{A_k}{z - p_k}
$$

系数由覆盖法得出：

$$
\boxed{A_k = \left[ (z - p_k) \frac{X(z)}{z} \right]_{z = p_k}}
$$

然后：

$$
X(z) = \sum_{k=1}^{N} \frac{A_k z}{z - p_k} = \sum_{k=1}^{N} \frac{A_k}{1 - p_k z^{-1}}
$$

**反变换**（每项 $\dfrac{A_k}{1 - p_k z^{-1}}$ 的反变换取决于 ROC）：

| 情况 | ROC | 反变换 |
|------|-----|--------|
| 因果（右边）序列 | $|z| > |p_k|$ | $A_k \, p_k^n \, u[n]$ |
| 反因果（左边）序列 | $|z| < |p_k|$ | $-A_k \, p_k^n \, u[-n-1]$ |

> 若未指定 ROC，通常默认 ROC 为最大极点模以右的区域（与因果序列对应）。

**示例**：求 $X(z) = \dfrac{1}{(1 - 2z^{-1})(1 - 3z^{-1})}$（$|z| > 3$）的反变换。

改写为 $X(z) = \dfrac{z^2}{(z-2)(z-3)}$，先求 $X(z)/z = \dfrac{z}{(z-2)(z-3)}$：

- $A_1 = \left[ \dfrac{z}{z-3} \right]_{z=2} = \dfrac{2}{-1} = -2$
- $A_2 = \left[ \dfrac{z}{z-2} \right]_{z=3} = \dfrac{3}{1} = 3$

$$
X(z) = \frac{-2z}{z-2} + \frac{3z}{z-3} = \frac{-2}{1 - 2z^{-1}} + \frac{3}{1 - 3z^{-1}}
$$

由 ROC $|z| > 3$（因果），反变换为：

$$
\boxed{x[n] = (-2 \cdot 2^n + 3 \cdot 3^n) u[n] = (3^{n+1} - 2^{n+1}) u[n]}
$$

---

#### 6.2.2 重极点

设 $X(z)$ 在 $z = p_1$ 处有 $r$ 重极点（其余为单极点）。对 $X(z)/z$ 展开：

$$
\frac{X(z)}{z} = \frac{A_{11}}{z - p_1} + \frac{A_{12}}{(z - p_1)^2} + \cdots + \frac{A_{1r}}{(z - p_1)^r} + \sum_{k=r+1}^{N} \frac{A_k}{z - p_k}
$$

系数由求导公式计算：

$$
\boxed{A_{1k} = \frac{1}{(r - k)!} \left[ \frac{d^{r-k}}{dz^{r-k}} \left( (z - p_1)^r \frac{X(z)}{z} \right) \right]_{z = p_1}}
$$

**反变换**（查表，因果情况）：

最常用的二阶重极点：$\dfrac{az}{(z - a)^2} \stackrel{\mathcal{Z}^{-1}}{\longleftrightarrow} n a^n u[n]$

**示例**：求 $X(z) = \dfrac{z(z+1)}{(z-1)^3}$（$|z| > 1$）的反变换。

令 $F(z) = (z-1)^3 \cdot \dfrac{X(z)}{z} = z + 1$：
- $C = F(1) = 2$，$B = F'(1) = 1$，$A = \frac{1}{2!} F''(1) = 0$

$$
X(z) = \frac{z}{(z-1)^2} + \frac{2z}{(z-1)^3}
$$

$$
\boxed{x[n] = n u[n] + 2 \cdot \frac{n(n-1)}{2} u[n] = n^2 u[n]}
$$

---

#### 6.2.3 复共轭极点

若 $X(z)$ 的系数均为实数，则复极点必定以共轭对出现：$p_{1,2} = r e^{\pm j\omega_0}$。

1. 对每个复极点单独使用覆盖法，求出复数系数 $A_1$ 和 $A_2 = A_1^*$
2. 利用欧拉公式合并为实数序列

将 $A_1$ 写为极坐标形式 $A_1 = |A_1| e^{j\phi}$：

$$
\boxed{x[n] = 2|A_1| \, r^n \cos(\omega_0 n + \phi) \, u[n]}
$$

**示例**：求 $X(z) = \dfrac{z^2 + z}{z^2 - z + 0.5}$（$|z| > 1/\sqrt{2}$）的反变换。

分母根：$z = \dfrac{1 \pm j}{2} = \dfrac{1}{\sqrt{2}} e^{\pm j\pi/4}$，因此 $r = 1/\sqrt{2}$，$\omega_0 = \pi/4$。

$$
A_1 = \left[ \dfrac{z + 1}{z - \frac{1-j}{2}} \right]_{z = \frac{1+j}{2}} = \dfrac{\frac{1+j}{2} + 1}{\frac{1+j}{2} - \frac{1-j}{2}} = \frac{1}{2} - j\frac{3}{2}
$$

$|A_1| = \dfrac{\sqrt{10}}{2}$，$\phi = \tan^{-1}(-3) \approx -1.249$

$$
\boxed{x[n] = \sqrt{10} \left( \frac{1}{\sqrt{2}} \right)^n \cos\!\left( \frac{\pi}{4} n - 1.249 \right) u[n]}
$$

---

### 6.3 幂级数展开法（长除法）

当 $X(z)$ 表示为有理函数时，直接用分子除以分母（长除法）可得到 $z^{-1}$ 的幂级数形式：

$$
X(z) = \sum_{n=-\infty}^{+\infty} x[n] z^{-n} = \cdots + x[-2]z^2 + x[-1]z^1 + x[0] + x[1]z^{-1} + \cdots
$$

**长除法的方向由 ROC 决定**：

| ROC | 序列类型 | 长除方向 | 商的排列 |
|-----|----------|----------|----------|
| $|z| > r$ | 因果（右边）序列 | 按 $z^{-1}$ 的升幂排列长除 | 降幂商：$x[0] + x[1]z^{-1} + \cdots$ |
| $|z| < r$ | 反因果（左边）序列 | 按 $z$ 的升幂排列长除 | 升幂商：$x[-1]z + x[-2]z^2 + \cdots$ |

**示例**（因果情况）：$X(z) = \dfrac{1}{1 - 0.5z^{-1}}$，$|z| > 0.5$。

长除：$1 \div (1 - 0.5z^{-1}) = 1 + 0.5z^{-1} + 0.25z^{-2} + 0.125z^{-3} + \cdots$

因此 $x[n] = (0.5)^n u[n]$。✓

> **适用场景**：长除法适合求序列的前若干项，或当 $X(z)$ 的形式不宜做部分分式展开时。但它不给出闭合表达式。

---

### 6.4 留数法

留数法是 z 反变换的一般方法，适用于任意形式的 $X(z)$。

**反演公式与留数定理**：

$$
x[n] = \frac{1}{2\pi j} \oint_{C} X(z) z^{n-1} dz
$$

利用留数定理：

$$
\boxed{x[n] = \sum_{\text{围道 } C \text{ 内的所有极点}} \text{Res}\left[ X(z) z^{n-1}, z = p_k \right]}
$$

**留数的计算**：

- **单极点** $z = p_k$：$\text{Res}[F(z), p_k] = \lim_{z \to p_k} (z - p_k) F(z)$
- **$r$ 重极点** $z = p_k$：$\text{Res}[F(z), p_k] = \dfrac{1}{(r-1)!} \lim_{z \to p_k} \dfrac{d^{r-1}}{dz^{r-1}} \left[ (z - p_k)^r F(z) \right]$

其中 $F(z) = X(z) z^{n-1}$。

**与部分分式展开的关系**：对于有理 $X(z)$，留数法与部分分式展开法等价——部分分式的系数正是 $X(z) z^{n-1}$ 在各极点的留数。

> **注意**：对 $n \leq 0$ 的情况，$z^{n-1}$ 可能在 $z = 0$ 处引入极点，须将原点处的留数也计入。

---

## 7. 单边 z 变换

### 7.1 定义与核心意义

**定义**（回顾 2.2 节）：

$$
\boxed{\mathcal{X}(z) = \mathcal{Z}_u\{x[n]\} = \sum_{n=0}^{\infty} x[n] z^{-n}}
$$

**为什么需要单边变换**：物理的离散系统中存在初始状态（如数字滤波器延迟单元中储存的历史值、内存状态等）。单边 z 变换通过在时移性质中引入 $x[-1], x[-2], \ldots$ 等初始条件项，自然地将这些信息纳入差分方程求解。

> **工程实践**：含初始条件的因果差分方程的求解几乎都使用单边 z 变换。

---

### 7.2 单边变换的时移与卷积性质

单边变换的时移性质与双边变换有显著不同——延迟操作引入了初始条件项。

**单位延迟**（$n_0 = 1$，最重要的情况）：

$$
\boxed{\mathcal{Z}_u\{x[n-1]\} = z^{-1} \mathcal{X}(z) + x[-1]}
$$

**证明**：

$$
\begin{aligned}
\mathcal{Z}_u\{x[n-1]\} &= \sum_{n=0}^{\infty} x[n-1] z^{-n} \\
&= \sum_{m=-1}^{\infty} x[m] z^{-(m+1)} \quad (m = n-1) \\
&= z^{-1} \left( x[-1] z^{1} + \sum_{m=0}^{\infty} x[m] z^{-m} \right) \\
&= x[-1] + z^{-1} \mathcal{X}(z)
\end{aligned}
$$

**单位超前**（$n_0 = -1$）：

$$
\boxed{\mathcal{Z}_u\{x[n+1]\} = z \mathcal{X}(z) - z x[0]}
$$

**一般延迟**（$n_0 > 0$）：

$$
\boxed{\mathcal{Z}_u\{x[n - n_0]\} = z^{-n_0} \mathcal{X}(z) + \sum_{k=0}^{n_0-1} x[k - n_0] z^{-k}}
$$

**物理关键**：与双边变换的简洁表达式 $z^{-n_0} X(z)$ 不同，单边变换中延迟一个样本额外引入 $x[-1]$ 项——这正是"延迟单元中存储的上一个值"。

**时域卷积**：对于两个因果序列 $x_1[n]$ 和 $x_2[n]$（$n < 0$ 时均为零），单边变换下的卷积性质与双边变换一致：

$$
\mathcal{Z}_u\{x_1[n] * x_2[n]\} = \mathcal{X}_1(z) \mathcal{X}_2(z)
$$

> 若参与卷积的序列之一非因果（在 $n < 0$ 处有非零值），单边卷积定理的形式将更复杂，通常需先将序列截取为因果部分再处理。

---

### 7.3 零状态响应与零输入响应

利用单边 z 变换求解离散 LTI 系统时，响应自然地分解为两部分：

$$
\boxed{y[n] = \underbrace{y_{zs}[n]}_{\text{零状态响应}} \;+\; \underbrace{y_{zi}[n]}_{\text{零输入响应}}}
$$

1. **零状态响应**（ZSR）：假设所有初始条件为零，仅由输入 $x[n]$ 激励产生的输出。$\mathcal{Y}_{zs}(z) = H(z) \mathcal{X}(z)$。

2. **零输入响应**（ZIR）：假设输入恒为零，仅由系统内部的初始状态（延迟单元中存储的历史值）产生的输出。

**叠加原理**：总响应 = ZSR + ZIR。在 z 域中表现为：

$$
\mathcal{Y}(z) = \underbrace{H(z) \mathcal{X}(z)}_{\text{ZSR}} \;+\; \underbrace{\frac{\text{由初始条件决定的多项式}}{A(z)}}_{\text{ZIR}}
$$

其中 $A(z) = 1 + a_1 z^{-1} + \cdots + a_N z^{-N}$ 是系统特征多项式（$H(z)$ 的分母）。

---

### 7.4 含初始条件的差分方程求解

**通用求解步骤**：

对于 $N$ 阶常系数线性差分方程：

$$
\sum_{k=0}^{N} a_k y[n - k] = \sum_{k=0}^{M} b_k x[n - k], \quad a_0 = 1
$$

给定输入 $x[n]$（$n \geq 0$）和初始条件 $y[-1], y[-2], \ldots, y[-N]$。

**步骤**：

1. 对两边取单边 z 变换，利用延迟性质代入初始条件
2. 整理得到关于 $\mathcal{Y}(z)$ 的代数方程
3. 解出 $\mathcal{Y}(z)$
4. 通过部分分式展开求 $\mathcal{Y}(z)$ 的反变换，得到 $y[n]$（$n \geq 0$）

**完整示例**：

求解：$y[n] - 0.5 y[n-1] = x[n]$，其中 $x[n] = u[n]$，初始条件 $y[-1] = 2$。

**第一步**：取单边 z 变换。

$$
\mathcal{Y}(z) - 0.5 \big[ z^{-1} \mathcal{Y}(z) + y[-1] \big] = \mathcal{X}(z)
$$

代入 $y[-1] = 2$，$\mathcal{X}(z) = \dfrac{1}{1 - z^{-1}}$（$|z| > 1$）：

$$
\mathcal{Y}(z) - 0.5 z^{-1} \mathcal{Y}(z) - 1 = \frac{1}{1 - z^{-1}}
$$

**第二步**：整理出 $\mathcal{Y}(z)$。

$$
(1 - 0.5 z^{-1}) \mathcal{Y}(z) = \frac{1}{1 - z^{-1}} + 1 = \frac{2 - z^{-1}}{1 - z^{-1}}
$$

$$
\mathcal{Y}(z) = \frac{2 - z^{-1}}{(1 - z^{-1})(1 - 0.5 z^{-1})}
$$

**第三步**：部分分式展开。

- $A = \left[ \dfrac{2 - z^{-1}}{1 - 0.5 z^{-1}} \right]_{z^{-1} = 1} = \dfrac{2 - 1}{1 - 0.5} = 2$
- $B = \left[ \dfrac{2 - z^{-1}}{1 - z^{-1}} \right]_{z^{-1} = 2} = 0$

$$
\mathcal{Y}(z) = \frac{2}{1 - z^{-1}}
$$

**第四步**：反变换（因果，$n \geq 0$）。

$$
\boxed{y[n] = 2 u[n]}
$$

**验证**：差分方程迭代——$y[0] = 0.5 \times 2 + 1 = 2$，$y[1] = 0.5 \times 2 + 1 = 2$，以此类推 ✓

**响应分解**：

- 零状态响应（$y[-1] = 0$ 时）：$y_{zs}[n] = 2(1 - 0.5^{n+1}) u[n]$
- 零输入响应（$x[n] = 0$ 时）：$y_{zi}[n] = 0.5^n u[n]$
- 总响应：$y[n] = 2(1 - 0.5^{n+1}) + 0.5^n = 2 u[n]$

---

## 8. LTI 系统的 z 域分析

### 8.1 系统函数（传递函数）$H(z)$

对于单位样值响应为 $h[n]$ 的离散 LTI 系统，输入 $x[n]$ 与输出 $y[n]$ 的关系由卷积和给出：

$$
y[n] = x[n] * h[n]
$$

对两边取 z 变换，由卷积定理得：

$$
Y(z) = X(z) H(z)
$$

由此定义系统的**系统函数**或**传递函数**（Transfer Function）：

$$
\boxed{H(z) = \frac{Y(z)}{X(z)} = \mathcal{Z}\{h[n]\}}
$$

**系统函数的三种等价定义**：

| 定义方式 | 表达式 | 含义 |
|----------|--------|------|
| 单位样值响应的 z 变换 | $H(z) = \mathcal{Z}\{h[n]\}$ | 基本定义 |
| 零状态响应与输入之比 | $H(z) = \dfrac{Y(z)}{X(z)}$ | 当初始状态为零时成立 |
| 特征函数对应的特征值 | $H(z) = \dfrac{y[n]}{x[n]}\big|_{x[n]=z^n}$ | 源于特征函数性质（第 1 章） |

**系统函数的代数结构**：

对于由常系数线性差分方程描述的离散 LTI 系统，$H(z)$ 总是一个 **z 的有理函数**：

$$
\boxed{H(z) = \frac{B(z)}{A(z)} = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}}
$$

**标准因式分解形式**：

$$
H(z) = K z^{N-M} \frac{(z - z_1)(z - z_2) \cdots (z - z_M)}{(z - p_1)(z - p_2) \cdots (z - p_N)}
$$

- $z_1, z_2, \ldots, z_M$：系统的**零点**
- $p_1, p_2, \ldots, p_N$：系统的**极点**
- $K = b_0$：**增益常数**

---

### 8.2 因果性、稳定性与极点位置

**因果性的 z 域判据**：

离散 LTI 系统是因果的 $\Longleftrightarrow$ $h[n] = 0$ 对 $n < 0$。

由于 $h[n]$ 是右边序列（因果序列），根据第 3 章的结论：

> **因果 LTI 系统的 ROC 是圆外区域 $|z| > r_{\min}$，其中 $r_{\min}$ 为 $H(z)$ 最大极点的模。**

此外，因果有理系统函数的分子阶数不大于分母阶数（$M \leq N$），否则 $H(z)$ 在 $z = \infty$ 处有极点，对应非因果的冲激导数项。

**稳定性的 z 域判据**：

离散 LTI 系统是（BIBO）稳定的 $\Longleftrightarrow$ $\displaystyle \sum_{n=-\infty}^{+\infty} |h[n]| < \infty$。

这一条件等价于单位圆落在 $H(z)$ 的 ROC 内：

> **稳定 LTI 系统的 ROC 包含单位圆（$|z| = 1$）。**

**因果性与稳定性的综合**：

将因果性和稳定性条件合并：ROC 为圆外区域且包含单位圆，意味着：

$$
\boxed{\text{因果稳定系统} \;\Longleftrightarrow\; \text{所有极点位于单位圆内部} \quad (|p_i| < 1)}
$$

**极点的稳定性分类**：

| 极点位置 | 时域模式（因果） | 稳定性 |
|----------|-----------------|--------|
| $|p| < 1$（单位圆内） | 衰减指数/衰减振荡 | **稳定** |
| $|p| = 1$（单位圆上，单极点） | 等幅振荡、阶跃 | 边界稳定（临界） |
| $|p| = 1$（单位圆上，重极点） | $n^k \times$ 振荡、斜坡 | **不稳定** |
| $|p| > 1$（单位圆外） | 增长指数/发散振荡 | **不稳定** |

**s 平面与 z 平面的稳定性对照**：

| 连续时间（s 平面） | 离散时间（z 平面） |
|--------------------|--------------------|
| 稳定 ↔ 极点实部 $< 0$（左半平面） | 稳定 ↔ 极点模 $< 1$（单位圆内） |
| 临界稳定 ↔ 极点在虚轴上 | 临界稳定 ↔ 极点在单位圆上 |
| 不稳定 ↔ 极点在右半平面 | 不稳定 ↔ 极点在单位圆外 |

---

### 8.3 差分方程描述的 LTI 系统

大多数离散系统由常系数线性差分方程描述。z 变换可将差分方程转化为代数方程。

**标准形式**：

考虑 $N$ 阶常系数线性差分方程：

$$
\boxed{\sum_{k=0}^{N} a_k y[n - k] = \sum_{k=0}^{M} b_k x[n - k]}
$$

其中 $a_0 = 1$（可总通过归一化达到），$a_k, b_k$ 为常数。

**零初始状态下的系统函数**：

假设所有初始条件为零，对方程两边取 z 变换，利用时移性质：

$$
\left( \sum_{k=0}^{N} a_k z^{-k} \right) Y(z) = \left( \sum_{k=0}^{M} b_k z^{-k} \right) X(z)
$$

由此直接读出系统函数：

$$
\boxed{H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-N}}}
$$

**物理直观**：
- 分母多项式（$a_k$ 系数）→ 反馈路径 → 极点
- 分子多项式（$b_k$ 系数）→ 前馈路径 → 零点
- $a_0 = 1$ 表示系统是因果可实现的（当前输出不依赖于未来的输入）

> 含初始条件的分析需使用单边 z 变换，详见第 7 章。