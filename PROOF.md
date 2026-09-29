# Speculative Decoding 严格 target 分布证明

## 1. 记号与算法

在给定历史 \(h\) 下，target 的条件分布为 \(p(\cdot\mid h)\)，draft 的真实 proposal 为 \(q(\cdot\mid h)\)。每轮 draft 至多提议 \(K\) 个 token：

1. 从真实 \(q\) 顺序采样 \(x_1,\dots,x_m\)。遇到 EOS，或达到剩余生成长度 \(r\) 时停止。
2. 用一次逻辑 target batch 评估候选位置对应的前缀分布 \(p_i=p(\cdot\mid h x_1\cdots x_{i-1})\)。若全部接受且仍有预算、且最后候选不是 EOS，再评估全部候选之后的 bonus 前缀。
3. 对每个候选使用独立均匀随机数，以
   \[
   \alpha_i(x_i)=\min\left(1,\frac{p_i(x_i)}{q_i(x_i)}\right)
   \]
   接受。若 \(p_i(x_i)=0\) 或 \(q_i(x_i)=0\)，则 \(\alpha_i(x_i)=0\)。
4. 首个在位置 \(j\) 拒绝时，丢弃 \(x_{j+1},\dots,x_m\)，输出一个修正 token，分布为
   \[
   r_j(y)=\frac{(p_j(y)-q_j(y))_+}{Z_j},
   \qquad
   Z_j=\sum_y (p_j(y)-q_j(y))_+.
   \]
   之后从新历史进入下一轮。
5. 若所有候选都接受：
   - 若候选末尾是 EOS，停止；
   - 若已经达到 `max_new_tokens`，停止；
   - 否则从 bonus 前缀的 target 分布额外采样一个 token。
6. `K=0` 时没有 draft 候选，只评估一个 target 位置并直接采样一个 token。

实现中的 query 使用“候选 token 之前”的前缀。例如位置 \(i\) 的 \(p_i\) 查询前缀是 \(h x_1\cdots x_{i-1}\)，不是包含 \(x_i\) 的前缀；bonus 才查询全部候选之后的前缀。

## 2. 单步概率守恒

固定历史 \(h\)。最终输出某个 token \(y\) 的概率由两类互斥事件组成：

- draft 提议 \(y\)，且接受；
- draft 提议某个 \(x\)，在 \(x\) 处拒绝，然后 residual 采样到 \(y\)。

因此

\[
\Pr[Y=y]
=
q(y)\min\left(1,\frac{p(y)}{q(y)}\right)
+
\sum_x q(x)\left(1-\min\left(1,\frac{p(x)}{q(x)}\right)\right)r(y).
\]

先计算拒绝事件的总质量：

\[
\begin{aligned}
Z
&=\sum_x q(x)\left(1-\min\left(1,\frac{p(x)}{q(x)}\right)\right)\\
&=\sum_x \max(0,q(x)-p(x)).
\end{aligned}
\]

因为 \(p\) 和 \(q\) 都归一化，

\[
\sum_y (q(y)-p(y))_+
=
\sum_y (p(y)-q(y))_+.
\]

所以拒绝总质量也等于

\[
Z=\sum_y (p(y)-q(y))_+.
\]

对接受项有

\[
q(y)\min\left(1,\frac{p(y)}{q(y)}\right)
=
\min(q(y),p(y)).
\]

修正项对所有提议 token 求和后与具体 \(x\) 无关：

\[
Z r(y)=(p(y)-q(y))_+.
\]

于是

\[
\begin{aligned}
\Pr[Y=y]
&=\min(p(y),q(y))+(p(y)-q(y))_+\\
&=p(y).
\end{aligned}
\]

因此单步输出严格服从 target 分布。

## 3. 零概率与无拒绝质量

- 若 \(q(x)=0\)，draft 不会提议 \(x\)，接受概率为 0；target 需要的概率质量只能由其他被拒绝提议的 residual 分支给出。
- 若 \(p(x)=0,q(x)>0\)，则 \(\alpha(x)=0\)，该候选一定拒绝；这正是“不能让零 target 概率 token 通过”的保护。
- 若 \(p=q\)，所有 \(p-q\) 的正部都是 0，此时 residual 为空，拒绝概率也为 0。空 residual 不应被采样；实现把“随机数拒绝但 residual 为空”视为运行时错误。
- greedy draft 是合法的确定性 proposal，但它没有全支持。例如 \(q(x)=\mathbf 1[x=g]\) 时，不能给其他 token 伪造 proposal 概率。若 target 在非 greedy token 上有质量，则 \(p(g)<1\) 时 greedy 候选以 \(1-p(g)\) 拒绝，residual 为该 token 以外的归一化 target 正差质量。

拒绝后不能“重新从 \(p\) 采样”。该做法会把接受分支已产生的 \(\min(p,q)\) 质量再加上整个 \(p\)，总输出质量大于 1 且分布错误。正确修正分支只补到 \(p-q\) 的正部。

## 4. 多候选与自回归延伸

对于候选链 \(x_1,\dots,x_m\)，draft 采样该链的联合 proposal 是

\[
\prod_{i=1}^{m}q_i(x_i).
\]

若前 \(j-1\) 个候选接受、第 \(j\) 个拒绝，则该事件对历史 \(h x_1\cdots x_{j-1}\) 的条件概率是

\[
\prod_{i<j}q_i(x_i)\alpha_i(x_i)
\cdot
q_j(x_j)(1-\alpha_j(x_j)).
\]

后续 \(x_{j+1},\dots,x_m\) 被丢弃；它们的 draft 随机选择以及已批量评估的 target 位置不进入输出分布，只影响计算量。由单步守恒，在位置 \(j\) 的边缘输出分布是 \(p_j\)，且与 proposal 随机数无关。因而修正后历史的下一个 token 条件分布仍是 target。

全部接受时，每个已接受位置的边缘分布为 \(p_i\)。bonus token 从全部接受后新前缀的 target 分布采样，不使用 proposal，因而 bonus 也保持 target 条件分布。

对任意有限输出序列 \(y_1,\dots,y_t\)，逐轮使用上述单步边缘性，并在每个新历史上重新条件化，可得

\[
\Pr[Y_1=y_1,\dots,Y_t=y_t]
=
\prod_{i=1}^{t}p(y_i\mid y_{<i},h).
\]

EOS 在接受或 residual 输出后立即终止，因此 EOS 后面不存在概率质量。剩余长度达到 `max_new_tokens` 时不再评估 bonus；这与直接 target 采样到长度上限停止完全一致。不同历史前缀必须分别调用条件概率函数，不能复用旧前缀的 \(p\) 或 \(q\)。

## 5. Temperature、top-k、top-p 的真实 target

本原型支持 temperature、top-k 和 top-p 组合，定义如下：

1. 先把模型权重归一化为基础分布 \(\tilde p\)。
2. \(T>0\) 时令
   \[
   p'_t\propto \tilde p_t^{1/T}.
   \]
   \(T=1\) 是恒等变换；\(T=0\) 定义为确定性 argmax，平局按词表顺序。
3. top-k 先按概率降序、词表顺序打破平局，保留前 \(k\)。
4. top-p 再在保留集合中按相同顺序保留累计概率直到达到阈值。
5. 对保留集合重新归一化，得到真正要保持的 target 分布 \(\bar p\)。

Speculative 保持的是这个“截断并归一化后”的 \(\bar p\)，不是模型原始权重。draft 也使用同一个变换后的真实 proposal \(\bar q\)；接受、拒绝和 residual 全部基于 \(\bar p,\bar q\)。测试使用 `Fraction` 对 top-p、top-k、组合截断以及逆温度为整数的 rational temperature（如 `T=1/2`，即平方后归一化）后的完整序列分布做精确核验。一般非整数逆温度的幂运算不适合精确有理数核验，属于浮点采样路径。

## 6. 计数含义

实现分别记录：

- `target_forwards`：逻辑 target batch 调用次数。一次调用可包含多个不同前缀。
- `target_positions`：实际向 target 请求并计算的条件分布数量，包含候选位置，以及全部接受后才会采样但已批量评估的 bonus 位置。
- `draft_forwards` / `draft_positions`：实际 draft 查询次数和位置数。
- `accepted_draft_tokens`：被接受的候选 token 数。
- `bonus_tokens`：全部接受后额外从 target 采样的 token 数。
- `correction_tokens`：首个拒绝后从 residual 输出的 token 数。
- `wasted_target_positions`：拒绝点之后已经批量评估、但本轮无法输出的 target 位置。

逻辑 forward 次数下降不等于硬件延迟下降。真实延迟还取决于 batch 效率、K、draft 成本、接受长度、kernel/调度开销、KV cache、拒绝时浪费的位置，以及生成长度/EOS 分布。

## 7. 为什么可能变慢

若 draft 很慢，每轮 draft 的 \(K\) 次顺序查询成本可能抵消 target batch 节省。若接受率低，典型接受长度接近 0，每个输出仍需要 target 调用，同时还付出了 draft 查询和后续候选 target 位置的成本。大 \(K\) 会增加每个 batch 的位置数，但拒绝点之后的位置被浪费；因此低接受率下较大的 \(K\) 可能使 `target_positions/output` 升高，而不是降低延迟。

`demo.py` 的低接受率例子展示了更多 rejection、更低平均接受长度以及更高的每输出 token target 位置数。

## 8. 精确核验与反例

测试不只依赖蒙特卡洛频率：

- `tests/test_speculative.py` 对小词表、短序列枚举所有输出序列。
- 概率使用 `Fraction`，逐条比较 speculative 枚举概率与直接 target 自回归枚举概率。
- 覆盖不同 target/draft 随机转移表、`K=0..3`、top-p、top-k、组合截断、不同历史前缀、零概率、EOS、长度上限和 temperature=0。
- `probability_with_wrong_residual` 允许替换 residual。演示明确展示“拒绝后直接从 target 重采样”和“uniform 修正”都会产生不同序列概率。

蒙特卡洛只作为补充。一次没有拒绝的轨迹不能证明算法等价；随机数、接受事件、拒绝修正和多轮历史都必须纳入概率核验。
