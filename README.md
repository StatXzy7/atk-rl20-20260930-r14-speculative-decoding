# atk-rl20-20260930-r14-speculative-decoding

严格保持目标自回归分布的 speculative decoding 原型（纯 Python 标准库，
不下载任何语言模型）。draft / target 都是可查询的有限词表条件概率。

## 运行

```bash
python3 demo.py          # 精确核验、计数表、反例、截断核验、变慢分析
python3 -m pytest -q     # 全部测试（含 Fraction 精确枚举）
```

## 结构

- `specdecode/probability.py`：接受比、residual、温度/top-p 变换、
  精确有理数采样（Fraction 拒绝采样）。
- `specdecode/models.py`：`RandomWeightsModel`、`TransitionTableModel`、
  退化 greedy 包装器，以及应用变换后策略的 `DistributionProvider`。
- `specdecode/speculative.py`：完整单轮算法（逐 token 接受、拒绝
  residual 修正、全接受 bonus、EOS、长度预算、$K=0$）与逻辑计数器；
  可注入故意错误的 residual 作为反例。
- `specdecode/enumeration.py`：独立精确枚举（不采样），对输出序列求
  Fraction 概率定律，并与直接 target 自回归定律逐序列比对。
- `specdecode/counterexamples.py`：错误 residual（拒绝后重采 target、
  uniform、未截断负值、proposal 不一致）与单 token 数值反例。
- `tests/test_speculative.py`：零概率/无拒绝质量/$K=0$/EOS/长度边界/
  后截断温度核验/反例/live 计数一致性/蒙特卡洛补充。
- `PROOF.md`：概率守恒、residual 推导、自回归扩展证明与计数口径。

## 关键结论

- 单步输出边际恒为 target：$\min(p,q)+[q-p]_+=q$，拒绝质量
  $Z=1-\sum\min(p,q)$。
- 精确枚举对任意 $K$ 给出与 target 完全相同的序列分布；
  top-p 核验针对**截断后**策略（并断言与原始分布不同）。
- 计数器是逻辑量：调用减少不等于硬件加速；慢 draft / 低接受率时
  投机解码会更慢（`demo.py` 第 6 节给出实测）。
