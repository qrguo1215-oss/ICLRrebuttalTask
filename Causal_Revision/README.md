# 因果质疑补实验：独立运行任务

此目录处理 reviewer 对因果解释的质疑。`../Steering_Revision/` 处理表格、样本集和报告数据修正，两项任务使用不同计划和输出目录。底层模型包装、向量、数据、原生 replay 和答案评测共用同一份代码，不需要先完成数据修正任务，也不要只下载本目录。

实现审计见 [AUDIT.md](AUDIT.md)，实际小规模验收和剩余限制见 [VALIDATION.md](VALIDATION.md)。

**目前交付的是代码与小规模验证，不是补实验结果。** 不预设结果支持论文；完整运行后仍可能否定部分假设。代码不会启动占用 GPU 的空转任务，不包含远端账号、密码或调度系统指令。

## 合作者快速运行

Linux、Python 3.10 或 3.11、支持 BF16 的 NVIDIA GPU。请克隆整个仓库。模型按固定 Hugging Face revision 自动下载到本目录 `models/`，并检查权重、tokenizer 和配置 SHA256；不依赖原作者机器路径。

```bash
git clone https://github.com/qrguo1215-oss/ICLRrebuttalTask.git
cd ICLRrebuttalTask/Causal_Revision
bash run.sh plan
bash run.sh smoke --gpus 0,1 --workers-per-gpu 1
bash run.sh run --gpus 0,1 --workers-per-gpu 1 --cpu-workers 4
```

`plan` 不加载模型、不启动 GPU，打印完整任务数。无参数执行也只显示计划。`smoke` 只使用每个配置的一道校准题，上限 256 token，绝不使用测试题；结果存在 `runs/smoke/`，不可混入正式结果。`run` 才运行正式实验，上限始终为 10000 token，默认输出 `runs/causal/`。

完成后把终端提示的 `runs/causal_results.tar.gz` 发回即可。里面包含原始生成、逐题判分、校准、协议与哈希、汇总、图片、盲评材料和运行日志，不包含模型权重。策略相似度部分还需真实盲评标签，见下文。

首次安装依赖需下载 PyTorch/CUDA 包；不要用 `--skip-install` 绕过版本不一致。需要另一个 Python 时加 `--python /path/to/python`。已有准确模型文件时可加 `--model-root /path/to/model-parent`，目录内须含 `DeepSeek-R1-Distill-Qwen-1.5B` 等原名子目录。代码会验证而非默默使用另一个 checkpoint。

## 算力、续跑与优先级

每个 worker 单样本生成，不改变 batch shape。默认每卡一个进程，避免按 GPU 型号猜测显存；显存有余量时可显式增加 `--workers-per-gpu 2`。脚本每 30 秒保存显存、利用率、温度和功耗。多进程提高吞吐但不保证特定利用率，增加并发不改变题目 RNG。数学评测在独立 CPU 进程池运行，不在 GPU 解码循环中做符号计算。

全部模块默认 140300 个生成任务（CPS 96900；SEAL/ReBalance 各 21700，未扣可复用记录），**请先看 `plan`，再安排时间**。这不是本次调试量。CPS/MATH500/1.5B 承担完整机制验证，SEAL 与 ReBalance 默认只复现 depth、replay、random 核心模块。可分阶段运行：

```bash
# 最优先：等次数、不同偏离深度；原生/token/thought/cache 分解
bash run.sh run --modules depth replay --gpus 0,1 --output runs/priority1
# 随机方向与反向对照
bash run.sh run --modules random --gpus 0,1 --output runs/random
# 独立 discovery seeds + item-level 交互 + 非 split 位置对照
bash run.sh run --configs math15_cps --modules occlusion --gpus 0,1 --output runs/occlusion
# 正确解答策略相似度的无 steering 对照
bash run.sh run --configs math15_cps --modules counterparts --gpus 0,1 --output runs/counterparts
```

重复执行**同一条命令**即可断点续跑。每个结果独立原子写入；worker 失败立即停下同阶段其他 worker，已完成文件保留。不同模块/配置组合使用不同输出目录，禁止改动同一个目录内的冻结协议。多个进程不能同时控制同一输出目录。进程退出不会留下模型 worker。

如数据修正任务已经完成，可加 `--revision-results ../Steering_Revision/runs/revision`。只有同代码签名、模型、数据、精度、运行环境和长度上限的 G/V/token replay/thought replay 才能导入，并保留原始来源哈希；不导入旧 sampling/calibration，不改写来源文件。结果可复用与不可复用的边界是显式检查，不靠文件名猜测。

## 固定协议与各模块实际检验什么

共同约束：冻结的 100 校准题、400 测试题；深因果分析使用原有固定 200 题。CPS/SEAL 使用各自原生 BF16 协议，ReBalance 使用其 FP16 与动态控制器。不同方法不能不经验证共用 greedy 结果。对每个题目清空模型缓存与 ReBalance 动态历史，影子分支同时复制 KV 并保存/恢复控制器。模型、向量、数据、代码和包版本均记入运行标识。

### 1. Depth：相同偏离次数是否一定产生相同收益？

在 100 校准题的**原始 baseline token prefix** 上重做前向扫描，以精确离散 Wasserstein-1 距离拟合 native deficit 分布，冻结 `(T, top_p)`。不访问正确性标签。deficit 为 baseline 最大 logit 减所选 token logit，先转 FP32 再作减法；token 变了但 deficit 为零的精度平票单独记录。

令参考 nucleus kernel 为 `q`，greedy token 为 `g`，`r=1-q(g)`。三种控制具有相同的 non-greedy 支持集和相同的偏离概率 `r`，只有条件权重不同：matched 保留原条件权重，shallow 使用 `T/2`，deep 使用 `2T`，但**不重新截断支持集**。matched 数学上复原原 kernel，而不是旧实验中额外添加未说明的混合概率。

包含两种互补实验。第一，在第 8 个真实 think/split frozen prefix，强制恰好一个 non-greedy token，随后全程 clean greedy；所有条件拥有相同位置和一次偏离，因此能直接检验偏离深度，不能被“采样次数更多”解释。没有第 8 个 split 或没有 alternative support 时对所有条件共同记为不可用，报告覆盖率，不按结果换题。第二，在整条新生成轨迹上反复应用三个 kernel，并增加官方参数 boundary sampling；共同 gating 随机数按 split ordinal 固定，但轨迹分叉后状态和长度不同，所以只称同状态同概率，不称全轨迹等次数。

默认完整生成与单点实验各 8 个 seed。`depth_audit.csv` 报告实际次数、条件 deficit、平票、覆盖率；`contrasts.csv` 报告成对差异。这里只检验深度校准贡献，不声称整个 CDF 精确匹配具有必要性，后者需要额外等均值/不同分布形状实验。

### 2. Replay：文本选择与隐状态路径

在共同 200 题上比较 G、V、token replay、完整 thought replay、cache-only 和三种 self 对照。token/thought replay 使用既有已审计实现：在同一当前 prefix 生成 native proposal，然后丢弃 donor KV 与动态控制器状态，沿 clean 分支重放文本。完整 thought 到下一个 split、think end、EOS 或全局 10000 上限结束，不额外截断到 128/2048 token。

cache-only 选择当前 clean logits 的 argmax，但提交同一状态下 native 分支的 KV 与控制器，后续继续自由生成。**不是强制重放整条原 baseline 文本**，因此缓存仍有机会影响未来 token。self-cache 做相同额外前向但两支都是自身；所有 self 结果必须逐 token 等于 G，否则汇总直接报错。

这个分解检验保留哪条计算通路足够/必要；不把 clean-cache 成功解释成排除了所有内部语义计算。

### 3. Random：方向是否有特殊性？

同一 200 题比较 native、反方向、8 个固定种子的同范数随机方向。CPS 额外提供每个方向的 label-free 行为尺度校准：在 100 校准题扫描 `0.25/0.5/1/2/4`，按每题 deficit 分布 W1 选 scale。保留全部方向，不依据正确率挑选方向或尺度。

同范数和行为匹配两组分别报告。八个方向只是明确的随机基准，不代表穷尽高维方向空间；报告每个方向，不把所有方向输出当作独立题目扩大显著性。

### 4. Occlusion：区分机会重叠与简单 headroom

CPS 在所有 400 题上比较 uncertainty-gated whole-thought sampling 与 sampling+native，阈值取校准题 margin 分布 q25/q50/全部。官方 sampling 为温度 0.6、top-p 0.95。增加相邻非 split 位置：在 split 后处理第一个非 split token 再开启 sampling，直到下一 boundary。两种位置分别用同一校准题自己的 margin 分位数，禁止按准确率调参。sampling+native 提交原生缓存，不是 clean-cache clone。

每个位置/强度有独立 discovery seeds 100–107 估计逐题 q；评估 seeds 默认 0–7，关键 q50 为 200–215。**不能使用同一批 seeds 估计 q 后再与含相同噪声的 `SV-S` 做相关**，也不能只挑 exploration 恰好失败的测试输出来统计 rescue。

报告固定 greedy 对/错分层、逐题额外收益、成对交互、救回与破坏。五折按题 cross-fit `r(1-q)-h*q` 的恒定 rescue/harm headroom 零模型：**仅用其他题目的 G/V 结果估计 r、h，再预测当前题 exploration 后的剩余收益，不用 S/SV 拟合自身效应**。分层汇报也保留完整训练折，避免错题子集无法估计破坏率。bootstrap 时重拟合，另报 seed 配对轮换敏感性及 discovery q 的 Wilson 区间。该检验只能拒绝“原生 rescue/harm 率可以原样迁移”这一明确受限零模型，不能排除所有 ceiling/distribution-shift 解释。非 split 对照不是自动的“等准确率/等计算”对照，实际探索次数和余量差异必须一起阅读。

### 5. Counterparts：93% 策略一致是否只是题目本来有标准解？

CPS 测试题 baseline 官方 token-level sampling 8 次，从首个生成 token 开始采样。对 G错/V对题，前 4 seeds 和后 4 seeds 为固定不交叠池。从前池取两个正确 candidate，分别比较 native 正确 query 和后池中的一个正确 baseline query。同题、同两个 candidate、相同 pool size，避免不同检索池造成优势；不足样本的题只报告覆盖不足，不偷偷增加采样或换题。此盲评样本还要求主评测与 strict 评测对正确性一致，评测分歧不混入“正确解答”池，并单独计入覆盖损失。

脚本产生 `counterparts/blind_packet.json`、标签模板与独立私有分组键。只把 blind packet 交给判读者；要求判断主要数学化简和关键推导，不以同答案或措辞相似代替策略一致。返回实际标签后按 README.txt 命令计分、更新打包。自动生成完成但未盲评时，不会宣称该科学结论已完成。

## 汇总与复核

```bash
# 只重做 CPU 判分/统计，不重新生成
bash run.sh report --output runs/causal --cpu-workers 8
```

结果检查包括完整主键覆盖、重复、100/400 边界、精度/运行环境、校准输入哈希、micro 同 prefix 与恰好一次偏离、self token 恒等。判分超时/异常单独保存并中止正式汇总，不能当作错题。主要评测保留历史 parser 规则，并列报告原始未裁剪输出、禁用隐式百分比容差的 strict sensitivity。MATH gold 从 `solution` 的 boxed answer 提取，GSM 使用已有 `answer`。

准确率差异按题配对 bootstrap 10000 次，先对每题的评估 seeds 取均值；另报逐 seed McNemar、seed 方差。置信区间跨零不等于等效。多条件结果是预注册探索性比较，不依据结果择优汇报。效率指标包括 token 数、EOS、达到长度上限、forward 次数与运行时；计时含 shadow/replay 和并发竞争，不能直接称作实际部署速度。

请保留返回包和 Git commit。需要补方法或模型时显式使用 `--configs`，并先重新执行 smoke。没有 7B CPS 向量，程序会拒绝该配置，不会冒充已有模型。
