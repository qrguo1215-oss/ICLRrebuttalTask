# 小规模验证记录

本文件记录代码验收，不是科学实验结果。没有运行正式 400 题测试、10000-token 全量任务，也没有生成任何论文结论。

## 已完成

- 21 项因果任务单元测试通过，16 项既有数据修正任务回归测试通过。
- 两张 NVIDIA A800-SXM4-80GB 上验证 CPS、SEAL、ReBalance；PyTorch 2.3.1+cu121、Transformers 4.46.3、NumPy 1.26.4、SDPA、TF32 关闭。
- 每种配置只使用同一道 calibration 题（index 5），每次最多 256 个生成 token；正式测试集完全未访问。
- 6 个配置/GPU 预检全部通过：零向量与禁用干预的 logits 最大差为 0；重复 native shadow 最大差为 0；输入 KV 和 controller 状态保持不变；跨问题 reset 正确；同 seed 采样在插入 native 运行后仍逐 token 一致，且跨 GPU 一致。
- 74 个计划项完整落盘。其中 70 个实际生成；ReBalance 的 4 个 micro 条件在同一个 frozen state 的参考 nucleus 支持内没有非 greedy 候选，**共同**记为不可用。没有扩大支持集、换题或把不可用计为错误。CPS/SEAL 的 micro 正向路径和该不可用保护路径均已覆盖。
- smoke 中 micro 使用第 1 个 boundary，以便短序列调试；正式第 8 个 boundary 的计数、think mask 和缺失分支由单元测试覆盖。不能把这称作第 8 个真实 boundary 的全量行为验证。
- 所有 self-cache、self-token、self-thought 都与对应 G 的 token 序列完全相同。
- CPU 判分、strict sensitivity、统计、CSV/图片、盲评材料、结果 tar.gz 打包均执行成功；判分异常/超时 0；结果 manifest 的所有文件 SHA256 验证通过。
- 盲评材料的等候选池、不交叠 query 池和返回标签计分用合成单元测试验证。没有虚构真实人工/LLM 标签。

## 修复与防回归

1. MATH 数据没有直接 `answer` 字段：补上从 `solution` 提取 gold，两个冻结数据集全部验证非空。
2. 旧单位清洗会吞掉 `\text{east}` 等文本答案：strict sensitivity 保留文本答案，并禁止空字符串相等被算作正确；主评测规则与 strict 结果并列保存。
3. 完整官方 sampling 从首个生成 token 开始；加入回归测试，不允许先 greedy 一个 token。
4. 主要 headroom null 的 rescue/harm 率来自其他题目折中的 G/V，而非用 S/SV 拟合要解释的效应。合成反例测试验证零模型不会把真实残差自动拟合掉。
5. 固定全部显式依赖版本，额外固定数学后端 mpmath；跳过安装也必须通过版本检查。

## 可核对标识

最终调试运行的生成代码/资产签名：

```text
d7636d7ba782e38769d059165f7de5754823d673a55ae58340bdf55569e86e08
```

最终 smoke run ID：

```text
6870a13ec30a8054b6be12eca525e66f81657cd6bf59af7f561f4eb896628d03
```

统计和评测使用独立的 `EVALUATOR.json` 哈希，不要求因 CPU 汇总修复而重跑 GPU。模型权重、环境、原始输出以及私有运行路径不提交到 Git；协作者需要在自己的硬件上重新 smoke，再启动正式实验。

剩余限制：本次未实测 7B、长至 10000 token 的全部条件或异构 GPU 混跑。短序列通过不能保证不存在长序列显存问题，默认每卡一个 worker，增加并发应由协作者依据显存决定。独立判读标签尚未产生，因此策略相似度结论仍待正式生成与真实盲评。
