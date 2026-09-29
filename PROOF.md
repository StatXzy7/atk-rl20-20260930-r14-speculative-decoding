# Speculative Decoding 严格保持目标分布：算法、证明与核验

本文给出一个只依赖可查询有限词表条件概率的 speculative decoding 原型，
严格证明其**逐 token 输出分布与 target 完全一致**，并用精确有理数枚举
独立核验。记号与代码一一对应。

## 1. 记号与前提

- 词表 $\mathcal V=\{0,\dots,V-1\}$，含一个可选终止符 `EOS`。
- 历史 $h=(t_0,\dots,t_{n-1})$ 是已确认 token 元组。
- target 条件分布 $q(\cdot\mid h)$；draft 条件分布 $p(\cdot\mid h)$。
  两者都是**真实**概率向量：$p_x,q_x\ge 0$，$\sum_x p_x=\sum_x q_x=1$。
- draft 允许是退化分布（greedy：某个 token 概率 1，其余 0）。它**不被
  假装成全支持 proposal**；所有公式显式处理 $p_x=0$。
- 每轮 draft 最多提议 $K$ 个 token；$K=0$ 是合法情形（纯 target 自回归）。
- target 在一轮内对候选前缀**一次批量评估**，对 $m$ 个 draft 候选加
  （可选的）一个 bonus 位置给出 $m$ 或 $m+1$ 个条件分布。

### 1.1 温度、top-k/top-p：到底保持谁的分布

温度与截断是**对策略本身的确定性变换**，不是采样噪声。本原型实现
温度 $T$ 与 top-p（nucleus）：

1. 原始非负权重 $w(x\mid h)$；
2. 温度：$w'_x = w_x^{1/T}$；
3. top-p：按归一化概率降序保留累计质量首次达到截断阈值 $\rho$ 的最小
   token 集合（跨过边界的 token 也保留），其余权重置 0；
4. 重新归一化得到 $\tilde q$。

**需要严格保持的 target 是第 4 步的后验策略 $\tilde q$，不是原始模型
分布。** 核验必须针对 $\tilde q$（代码中 `DistributionProvider.prob`，
测试 `test_top_p_verifies_post_transform_law` 同时断言与后截断定律
相等、与原始定律不相等）。draft 侧 acceptance 使用 draft 的**真实**
proposal 概率：若 draft 未截断，就用未截断的 $p$，二者是两个独立策略。

精确有理数模式下温度只接受 $T=1/n$（$n$ 为整数，例如 $T=\tfrac12$），
因为 $w^{1/T}$ 必须仍为有理数；一般实数温度在 float 模式下可用，
此时等价性由蒙特卡洛补充验证。

## 2. 单轮算法

设当前历史为 $h$，剩余生成长度预算为 $R$。

1. **Draft 阶段（串行）。** 令 $h_0=h$。对 $j=0,1,\dots$，当
   $j<K$ 且仍有非终止预算时，从 $p(\cdot\mid h_j)$ 采样
   $y_j$，令 $h_{j+1}=h_j\circ(y_j)$；若 $y_j=\mathrm{EOS}$ 立即停止。
   记得到 $m$ 个 draft token（$0\le m\le K$）。
2. **Target 一次批量评估。** 计算
   $q^{(j)}=q(\cdot\mid h\circ y_{<j})$，$j=0,\dots,m-1$。
   若最后一个 draft 不是 EOS 且长度预算未满，再计算
   $q^{(m)}=q(\cdot\mid h\circ y_{<m})$ 用于 bonus；
   否则不评估 bonus 位置。
3. **逐 token 接受。** 对 $j=0,\dots,m-1$，以概率
   $a_j=\min\!\bigl(1,\,q^{(j)}_{y_j}/p(\,y_j\mid h_j)\bigr)$
   接受 $y_j$（独立 Bernoulli）；一旦出现拒绝，设拒绝位置为 $i$，
   跳到步骤 4。若 draft 以 EOS 结束且全部接受，生成结束。
4. **拒绝修正。** 在拒绝位置 $i$ 从 residual 分布采样 $x$：
   $$r(x)=\frac{[q^{(i)}_x-p_i(x)]_+}{Z},\qquad
     Z=\sum_x[q^{(i)}_x-p_i(x)]_+,$$
   其中 $p_i(x)=p(x\mid h\circ y_{<i})$，$[u]_+=\max(0,u)$。
   输出前 $i$ 个已接受 draft token 与 $x$，以
   $h\circ y_{<i}\circ(x)$ 为新历史进入下一轮。
5. **全部接受 + bonus。** 若 $m$ 个 draft 全部接受且存在 $q^{(m)}$，
   从 $q^{(m)}$ 直接采样一个 bonus token $x$（这是纯 target 采样，
   不需要 draft 提议）；输出 $y_{<m}\circ(x)$。若 bonus 为 EOS 或
   长度用尽则停止，否则以新历史进入下一轮。

零概率规则（代码 `accept_prob`）：

- $p_i(x)=0$：draft 永远不会提出 $x$，不存在“接受 $x$”事件，定义
  接受比为 0；
- $q^{(i)}_x=0$ 且 $p_i(x)>0$：提出时必拒，接受比为 0；
- $p_i(x)=q^{(i)}_x=0$：该 token 在该历史下不可能出现，无贡献。

$Z=0$（无拒绝质量）意味着 target 被 draft 处处随机占优
（$q^{(i)}_x\le p_i(x)$ 对所有 $x$），拒绝事件总概率为 0，
**不允许** residual 采样；若代码在此处真的到达采样，说明实现有误，
直接抛出异常（`residual_distribution` 返回 $Z=0$，驱动检查）。

## 3. 单步概率守恒证明

固定一个历史 $h$（省略其上标），设 draft 一步提出 $x$（$p_x>0$）。
$x$ 被**输出**的概率是“接受 $x$”与“拒绝并从 residual 修正为 $x$”
两条互斥路径之和。

**接受路径。** 提出 $x$ 的概率 $p_x$，接受概率 $\min(1,q_x/p_x)$，故
$$\Pr(\text{输出 }x\text{ 且来自接受})
  =p_x\min(1,q_x/p_x)=\min(p_x,q_x).$$

**拒绝路径。** 记总接受质量
$$\alpha=\sum_z\min(p_z,q_z),$$
则拒绝事件概率为 $1-\alpha$。由恒等式
$$1-\sum_z\min(p_z,q_z)
 =\sum_z(q_z-\min(p_z,q_z))
 =\sum_z[q_z-p_z]_+=Z,$$
拒绝分支修正为 $x$ 的概率为
$$(1-\alpha)\,r(x)=Z\cdot\frac{[q_x-p_x]_+}{Z}=[q_x-p_x]_+.$$

因此
$$\Pr(\text{输出 }x)
 =\min(p_x,q_x)+[q_x-p_x]_+.$$
按 $q_x\le p_x$ 与 $q_x>p_x$ 两种情形：

- $q_x\le p_x$：$\min(p_x,q_x)=q_x$，$[q_x-p_x]_+=0$，和为 $q_x$；
- $q_x>p_x$：$\min(p_x,q_x)=p_x$，$[q_x-p_x]_+=q_x-p_x$，和为 $q_x$。

故对所有 $x$，
$$\boxed{\Pr(\text{该位置输出 }x)=q_x.}$$

归一化作为附带结论：
$\sum_x\min(p_x,q_x)+Z=\sum_x q_x=1$，修正分支是良定义的概率分布。

**禁止做法的失败原因。** 若拒绝后“直接重新从 target 采样”（反例
`renorm_target_residual`），边际变为
$$\Pr(x)=\min(p_x,q_x)+(1-\alpha)q_x\ne q_x,$$
因为接受阶段已经消耗了一部分 $q_x$，再按 $q_x$ 分配拒绝质量属于重复
计数。显式数值反例：$q=(\tfrac12,\tfrac12)$，greedy draft
$p=(1,0)$。$\alpha=\tfrac12$，$Z=\tfrac12$，正确 residual 为
$(0,1)$，边际 $=(\tfrac12,\tfrac12)=q$；错误重采样给出边际
$(\tfrac34,\tfrac14)$，TV 距离 $\tfrac14$。

**greedy 退化 proposal 不违反任何条件**：证明从不要求 $p_x>0$；
$p_x=0$ 的 token 不产生提议事件，其 target 超出质量全部通过 residual
输出。错误的是把 greedy 向量在实现里误当成平滑/全支持向量使用。

## 4. 多 token 轮与 bonus 的分布

一轮内 draft 序列 $\mathbf y=(y_0,\dots,y_{m-1})$ 的提出概率为
$\prod_j p_j(y_j)$。前 $i$ 个接受、第 $i$ 个拒绝的联合概率为
$$\prod_{j<i} p_j(y_j)\min(1,q^{(j)}_{y_j}/p_j(y_j))
 \cdot p_i(y_i)\bigl(1-\min(1,q^{(i)}_{y_i}/p_i(y_i))\bigr).$$
对所有 draft 序列求和并乘以 residual 概率后，在第 $i$ 个输出位置的
边际与第 3 节完全相同（前面已接受的 token 逐位都等于对应 target 条件
分布，拒绝位等于 residual 修正）。

全部接受时 bonus 直接从 $q(\cdot\mid h\circ\mathbf y)$ 采样，
因此 bonus 条件分布就是 target。关键的一点是**条件化的历史必须是真实
被接受的前缀**：下一轮 target/draft 查询都使用
$h\circ y_{<i}\circ(x)$ 或 $h\circ\mathbf y\circ(x)$，被丢弃的 draft
尾部绝不进入历史（驱动与枚举都严格按此查询）。

## 5. 自回归序列层面的保持

用归纳/链式法则论证。令 $T_n$ 表示算法输出前 $n$ 个 token 后的已确认
历史随机变量。

- **基例**：$T_0$ 为给定 prompt，确定。
- **归纳步**：假设 $\Pr(T_n=h)=\prod_{t<n}q(t_t\mid t_{<t})$，
  即历史分布与 target 自回归完全一致。给定 $T_n=h$，一轮内输出的
  每个新 token（第 4 节）条件边际都是 $q(\cdot\mid h')$，其中 $h'$ 是
  该 token 真实的条件历史；多输出一轮只是连续应用该单步结论。
  因而对任意长度 $\ell$ 的续写 $u_{<\ell}$，
  $$\Pr(u_{<\ell}\mid T_n=h)=\prod_{j<\ell}q(u_j\mid h\circ u_{<j}).$$
  对 $h$ 按归纳假设求边缘即得 target 联合分布。
- EOS 是普通 token：$q(\mathrm{EOS}\mid h)$ 决定终止概率；一旦输出
  EOS 立即停止，与 target 自回归采样的停止规则一致。长度上限
  $L$ 只是把“未遇 EOS 的条件续写”截断在 $L$，两边使用同一规则。

**结论：** 对任意 prompt、任意 $K\ge 0$、任意（可退化的）draft
策略、是否启用温度/top-p，算法对完整输出序列的联合分布恒等于
$$\prod_{j} q_T(u_j\mid \text{prompt}\circ u_{<j}),$$
其中 $q_T$ 是（可能经温度/截断变换后的）真正 target 策略。$K$ 只改变
效率，不改变分布；这正是代码精确枚举在 $K=0,1,2,3$ 下得到同一定律的
原因。

## 6. 边界情况处理

- **零概率**：proposal/目标零质量按第 2 节规则处理；枚举自动跳过
  质量为 0 的 draft 路径，`path_total == 1` 断言保证无遗漏。
- **无拒绝质量 $Z=0$**：拒绝分支概率为 0，不做 residual 采样；
  测试中同策略（target=draft）与确定性 target 都覆盖此情形，
  精确期望 correction token 数为 0。
- **$K=0$**：不做任何 draft 查询，一轮评估一个 target 位置并把它作为
  “bonus”输出，即普通自回归；计数上 `draft_queries=0`，
  每次 target 调用恰好 1 个位置。
- **EOS**：draft 采样到 EOS 立即停止继续提议；target 不再为其评估
  bonus 位置；EOS 只能作为一轮/序列最后 token。
- **剩余长度**：draft 候选数取 $\min(K, R-1)$（有 EOS 时为必须保留的
  终止位置；无 EOS 时为 $\min(K,R)$），bonus 仅在预算未满时评估。
  输出长度永不超过 $L$。
- **不同历史前缀**：target 与 draft 的每次查询都传入完整已确认历史；
  被拒 draft 之后的候选前缀评估值只用于计数，不用于条件化下一轮。

## 7. 计数口径（逻辑量，非硬件指标）

驱动记录：

| 计数器 | 含义 |
|---|---|
| `draft_queries` | draft 串行条件查询次数（提议 token 数） |
| `target_calls` | target 逻辑批量评估轮数 |
| `target_positions_requested` | target 批量中实际评估的 token 位置数（含拒绝后丢弃的尾部与未用 bonus） |
| `target_positions_useful` | 产生了被保留 token 的位置数；期望上恒等于朴素自回归 target 步数 |
| `accepted_draft_tokens` | 被接受的 draft token 数 |
| `correction_tokens` | residual 修正 token 数 |
| `bonus_tokens` | 全接受后额外 target token 数 |

每轮 `requested = m (+1 bonus)`，`useful = 拒绝位置 i+1` 或
`m(+1 bonus)`；二者之差就是“算了但没用”的位置。**逻辑调用次数减少
不自动等于加速**：真实墙钟时间取决于批量 target 对 $K+1$ 个位置的
并行代价、$K$ 次串行 draft 代价以及接受率。演示中把每次 draft 查询
人为拖慢 1.5 ms，投机方案显著慢于 $K=0$ 基线。粗略条件：
$$c_T^{\text{batch}}(K+1)+K c_D
 <\frac{\text{每轮期望输出 token 数}}{1}\,c_T,$$
接受率低时右侧变小、左侧（含丢弃位置）不变甚至浪费增大。

## 8. 独立核验方法

1. **精确枚举（证明的可执行版本，`specdecode/enumeration.py`）**：
   全程使用 `Fraction`，对每个 draft 路径、每个 Bernoulli 接受结果、
   每个 residual/bonus token 求联合概率并递归，对小词表短序列穷举所有
   输出序列；独立的 `exact_target` 直接按 target 链式法则求参考定律。
   二者字典逐序列严格相等（无容差）。
2. **期望计数**：枚举同时给出各计数器的精确期望，并断言
   `target_positions_useful` 与朴素自回归一致。
3. **反例**：错误 residual（重归一化 target、uniform、未截断负值、
   提议/修正使用不同 $p$）在精确枚举下必然产生不同定律。
4. **蒙特卡洛（仅补充）**：live 驱动数万次采样的频率只在抽样噪声内
   逼近精确概率；一次“没遇到拒绝”的运行不能证明任何等价性。
