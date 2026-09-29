# Speculative Decoding 原型

纯 Python 标准库实现，不下载语言模型。draft/target 是可查询的有限词表条件概率函数，支持小转移表和随机有限模型。

## 运行

```bash
python3 -m unittest discover -s tests -v
python3 demo.py | tee demo_results.txt
```

## 代码

- `specdec/distributions.py`：归一化、temperature/top-k/top-p、接受概率、residual。
- `specdec/models.py`：确定性表模型、精确分数随机模型、浮点运行时随机模型。
- `specdec/sampling.py`：逐 token 接受、拒绝修正、bonus、EOS/长度边界和计数。
- `specdec/exact.py`：用 `Fraction` 枚举完整输出序列分布，并支持替换错误 residual。
- `tests/test_speculative.py`：精确等价、零概率、K=0、EOS/长度、截断、greedy proposal 和错误 residual 反例。
- `demo.py`：精确核验、错误 residual 展示、蒙特卡洛补充和计数结果。
- `PROOF.md`：概率守恒证明、自回归延伸、边界条件、计数和变慢原因。

## 当前计数结果

`demo_results.txt` 保存了一次 40,000 样本运行结果。核心数值：

- 精确枚举：speculative 与直接 target 分布完全相等，总质量均为 1。
- 正常示例：logical target forwards/output = 0.5328，target positions/output = 1.3294，平均接受长度 = 1.1058。
- 低接受率示例：rejection rate/round = 0.7437，target forwards/output = 0.6587，target positions/output = 1.5213。

forward 计数减少只是逻辑调用聚合，不自动表示硬件加速。
