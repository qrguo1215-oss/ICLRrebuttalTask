# 发布前历史残留核查

本任务不是将原工作区脚本直接压缩。原代码依赖大型实验目录、固定模型路径、不同历史 seed 映射和外部判分模块；这些依赖已拆成独立的最小包。旧实验和仓库内其他任务没有被改写。

## 必须在生成前修正

**跨题状态污染。** ReBalance 的 `start_new_round` 不清空全部置信度历史，旧 direct scan 甚至只在 worker 启动时配置一次向量。新的 scan 与 generation 共用 `Engine.reset`，同时清空 `_coefs`、累计置信度、token 计数、上一词置信度及 `_prev_step_mean`。每个反事实分支独立保存/恢复动态状态与 KV。

**旧校准不能直接代表新实现。** 旧 grouped-softmax 计算和真正执行的 token-level FP32 top-p 有数值路径差异。新校准与生成共用 `native.nucleus`，先在完整词表计算概率及核支持，再聚合等 deficit。W1 是这些离散分布的精确分段积分，不用直方图近似，不采样近似概率。

**修正范围。** 所有十个有效配置在各自原100题、原候选网格上重做固定前缀校准。大部分网格130点，GSM 7B ReBalance 保留66点细化网格。向量、干预位置、模型、dtype、目标函数、问题权重不改变，不使用测试正确率。旧 selected/W1 仅保留为参考，不能冒充新扫描结果。

**校准输入。** 使用 baseline 保存的 token IDs；不再从删掉特殊 token 或被截断的文本重新分词。只接受明确的 calibration100 索引，核对基线 token 哈希、配置、候选网格、代码、模型、环境和运行 ID。无法读取完整证据时停止，不把两侧都缺失的哈希当成匹配。

**平票含义。** deficit=0 不一定表示 token 未改变：另一个并列最大 token 的 deficit 也是0。CDF 保留点质量；报告分别给出 positive-deficit rate 和 token-change rate。

**距离精度。** baseline logits 先转FP32再做两侧的减法，避免native侧BF16/FP16先减再转型、sampling侧FP32减法导致W1使用两套距离坐标。单元测试包含低精度反例。

## 结果一致性

正式任务共享唯一的 G/V 记录，避免表2/6/7/图6各算出一套基线。共用 baseline 只发生在数学上相同的 SEAL/CPS BF16无干预配置；ReBalance FP16 单列。表2始终固定200题，其他正式统计400题。7B CPS不存在有效向量，因此不构造该配置。

S/S+V 必须有八个真实 seeds；whole-thought 保留其原 seed 数。默认仅补 GSM whole-thought 的100/400修正，MATH internally consistent 部分不自动重跑。没有任何 synthetic/placeholder 结果作为输入。

完整 thought 在下一 split、think end、EOS 或全局10000-token上限结束，不再有128/2048段长上限。全局上限截断单独记录。S+V/E+V 保留 native cache，只有 replay 丢弃 donor cache。

## 判分审计

原数学判分器有隐式×100/÷100容差，原输出清理也可能删掉 `Question:`/`Comment:` 之后的有效答案。为了不悄悄改变历史指标，主列保留原口径，并对完整响应增加严格敏感性判分。分歧全部导出，发表前人工核对。超时/程序错误有独立状态，不自动记为答错，不生成不完整统计；重试只消耗CPU，不重新生成。

判分依赖整体参与独立的 evaluator 身份计算，不与生成身份混在一起。修复判分/报告不要求重新GPU生成；旧评分缓存会因evaluator身份变化而失效。原先 `Queue.get_nowait` 的瞬时队列可见性问题改为 Pipe 传输与显式超时；判分子进程从独立CPU进程创建，避免从CUDA进程 fork。

## 跨机器与恢复

模型使用相对缓存目录并自动下载。两个模型均固定Hub提交，并通过每个模型权重、tokenizer和config的预登记SHA256，内容漂移立即停止。额外权重或tokenizer覆盖文件会被拒绝，防止Transformers优先加载未校验的文件。因此模型身份不依赖目录名、文件mtime或当前Hub的“最新版”。

结果身份不包含部署目录或GPU编号，但包含代码、资产、模型内容、dtype、CUDA/PyTorch/Transformers/NumPy版本、GPU架构和固定seed映射。可以更改GPU数量和每卡worker数后恢复，不能混用改变协议或数值环境的记录。

每题原子写入、主控制进程排他锁、严格覆盖检查；worker失败时终止同阶段未完成worker，已完成结果保留。正式执行前强制smoke，仅使用校准题，不再用heldout200调试。没有数据驱动的“调到结果符合假说”为止的自动逻辑。

主进程捕获SIGTERM/SIGHUP并清理worker；worker启用Linux父进程死亡信号，避免控制进程被强杀后继续写同一结果。进程被杀时留下的临时文件不会被当成完成记录。

## 不保证的事项

这些检查不能保证某个科学结论复现，也不保证跨硬件逐token一致；它们确保不同结论不会被遗漏、混合协议或伪造区间掩盖。GPU cumsum 在所用PyTorch版本中只能warn-only，此例外明确记录。所有正式结果仍必须由完整生成和判分产生。
