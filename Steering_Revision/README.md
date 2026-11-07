# Steering Revision：论文结果修正补跑

这是一个独立任务，不依赖另一目录的 CPS 构造实验。**合作者只需运行下面的命令，完成后把结果压缩包发回。** 不需要修改路径、选择向量、重新划分数据或手工拼接结果。

## 快速开始

环境：Linux，Python 3.10/3.11，支持 BF16 的 NVIDIA GPU。建议每个并行 worker 至少 24 GB 显存；7B 长上下文/replay 建议 40 GB 以上。需能访问 PyPI 和 Hugging Face，首次自动下载两个模型，预留至少 40 GB 模型空间及足够的结果空间。

```bash
git clone https://github.com/qrguo1215-oss/ICLRrebuttalTask.git
cd ICLRrebuttalTask/Steering_Revision
./run.sh --gpus 0,1,2,3
```

`--gpus` 改成实际可用的 GPU 编号。程序自动创建本地环境、安装固定依赖、下载并校验模型、执行前置测试，然后运行正式任务、CPU 判分、统计和打包。

**中断后原命令重启即可。** 已完成的逐题结果不会重跑；代码、环境或校准不一致时会拒绝混用，不能通过手工改文件绕过。

完成后返回：`runs/revision_results.tar.gz`。其中包含逐题生成、真实 seed、校准/CDF、判分状态、表 2/6/7/8 和 GSM 图 6 的数据、置信区间及校验清单。模型权重不在结果包内。

## 先检查成本或环境

```bash
./run.sh plan                      # 只列任务量，不下载、不推理
./run.sh verify                    # 只检查随仓库提供的数据/向量
./run.sh smoke --gpus 0,1           # 小样本全流程检查，不启动正式实验
./run.sh --gpus 0,1 --workers-per-gpu 2
```

默认每卡一个独立、batch-size=1 的 worker。80GB 卡可用两个 worker 提高利用率；较小显存请保持默认。OOM 会停止该阶段并保留已完成结果，减少并发后原输出目录可继续。不要修改 batch size、精度或长度来规避 OOM。

默认完整任务为90000条生成和1000题固定前缀扫描，不是短演示。`plan` 可先核对任务规模；每30秒记录指定GPU的利用率、显存、温度和功耗，保存在 `logs/gpu.jsonl`。预检只使用校准题，不能将其短回答当作正式结果。

可选参数：`--model-root models`、`--output runs/revision`、`--cpu-workers 4`。相对路径从本任务目录解析；也可指向已有完整模型目录的父目录，内容仍必须通过校验。需要镜像时可自行设置 `HF_ENDPOINT`，不改变模型内容检查。

## 这次实际补什么

| 对象 | 固定协议 |
|---|---|
| 表 2 | MATH500 共用既有固定200题；五个真实模型/方法组合的 G、V、token replay、完整 thought replay。 |
| 表 6/7 | 两个模型 × 两个数据集，SEAL/ReBalance/CPS；7B CPS 缺有效向量，不运行、不填值。每数据集 calibration100/test400，S/S+V 为真实 seed 0–7。 |
| 表 8 | 原候选 T/top-p 网格，原100题，原 problem-equal W1 目标；用修正后的同前缀前向重新计算，再冻结参数。**不按正确率调参。** |
| GSM 图 6 | 改为100题校准、400题测试。q10/q25/q50/ALL，E/E+V，保留既有 seed 0。 |
| MATH 图 6 | 原实验内部一致，不默认重跑；需要时显式 `--scopes replay,direct,whole` 才追加。不能将默认结果包声称为整张图6重跑完成。 |

模型仍为 `DeepSeek-R1-Distill-Qwen-1.5B/7B`。原生 SEAL/CPS 使用 BF16、ReBalance 使用 FP16。向量、层、系数和方法 wrapper 没有替换为统一的近似算法。所有正式生成最多10000新 token；没有局部 thought 截断。采样只作用于各方法原有的双换行 boundary，whole-thought 条件才持续到下一 boundary/think end。

G/V 只生成一次，表2取同一结果的200题子集，表6/7和GSM图6取400题。CPS 与 SEAL 的无干预 BF16 baseline 共用；ReBalance 的 FP16 baseline 单独生成。不能跨精度拼基线。八 seed 只属于 S/S+V，不能把 whole-thought 的单 seed 写成八 seed。

正式补跑默认范围为 `replay,direct,gsm-whole`。可用 `--configs math15_seal,gsm15_seal` 分批承担配置；这会形成独立运行包，不能未经环境/协议校验混成一个结果。

## 防止昂贵返工的检查

1. 冻结数据索引和向量随仓库提供；模型通过完整文件 SHA256 校验，不仅看名称或文件大小。
2. ReBalance 每题清空所有动态历史；影子分支隔离 KV 和控制器状态。首次运行必须通过 null/self-replay、同 seed、影子读取和跨题历史不变性检查。
3. 新校准使用保存的原始 token IDs，不从清理后的文本重新分词。基线和 native 读数来自同一次固定 prefix。
4. 校准与正式采样调用同一个 token-level FP32 softmax/top-p 实现，保留零 deficit 的平票质量。正 deficit 比例与 token-change 比例分开记录。
5. 所有方法都在原100题、原网格重新计算 W1。旧值仅作参考，避免旧状态残留、浮点 cutoff 差异和新实现混用；这没有增加实验假设或用测试正确率挑参数。
6. 正式统计必须覆盖全部预注册 `题目 × 条件 × seed`；不会用合成数值补缺，也不会把负点估计写成显著负效应。

前置测试只能降低风险，不能保证不同硬件逐 bit 复现原数值；同一运行包会强制核对推理环境和 GPU 架构，避免悄悄混用。

## 判分与结果解释

主报告保留原判分口径以便对照旧实验，同时额外计算“不截断响应、关闭隐式百分比倍数容差”的敏感性结果。所有差异单列 `grading_sensitivity.json`，发表前需要核查，不会悄悄改答案标签。超时/程序错误不算普通答错：流程停止统计，保留 GPU 生成，只需修复后重试 CPU `report.py`。

```bash
.venv/bin/python report.py --output runs/revision --cpu-workers 4
```

`summary/` 包含 CSV 和机器可读汇总。准确率使用实际 seed 的均值/标准差；效应区间按题目平均 seeds 后做10000次配对 bootstrap。调试输出独立保存，正式统计拒绝接收。

本任务没有远端登录、HPC 调度、密码、访问令牌或 GPU keepalive。运行完成或失败后退出并释放资源。完整技术审计见 [AUDIT.md](AUDIT.md)，来源与许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

已执行的测试及其边界见 [VALIDATION.md](VALIDATION.md)。这份仓库提供修正后的实验实现，不预先承诺任何准确率或机制结论。
