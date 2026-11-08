# 实现审计与结论边界

本任务不修改旧生成文件，不重用未校验的历史离线统计。以下事项是代码验收标准，不是预期实验结论。

| 风险 | 实现与检查 |
|---|---|
| 校准题/测试题混用 | 冻结 assets split；smoke 仅一题 calibration；正式 100/400 |
| 不同方法偷偷共用基线 | 默认各自 G；仅严格签名和精度一致的修正结果可导入 |
| 旧 kernel 与新公式混用 | frozen forward 重校准；不导入任何旧 sampling 结果 |
| 所谓等次数只是相同概率 | micro 强制一次；live 显式报告实际次数且禁止精确等次数主张 |
| 深度控制同时改变候选支持 | shallow/matched/deep 共用原 nucleus 支持，只改条件权重 |
| BF16 平票被当作无干预 | 记录 token identity 与 FP32 logit deficit 两项 |
| dynamic controller 跨问题污染 | 每题 reset，包括 ReBalance `_prev_step_mean` |
| counterfactual 修改主缓存 | shared shadow clone + controller snapshot/restore；GPU no-op audit |
| cache-only 被强制文本掩盖 | 仅本步 clean token，native KV 推进，后续自由生成；toy 回归测试 |
| thought 人为截短 | 复用完整 thought proposer，不设局部长度 cap |
| q 与差值共享采样噪声 | 独立 discovery/evaluation seed 区间 |
| 历史两个 seed 批次择优 | 新协议同批 16 个 q50 seeds，明确不是解释历史批次来源 |
| ceiling 模型被过度解释 | 只检验 cross-fit constant rescue/harm null；输出配对敏感性 |
| 相似策略的高基准率 | 同题、等候选池、baseline correct-correct 盲评对照 |
| 找不到样本就换题 | 记录 unavailable 和覆盖率，不按结果补抽 |
| 生成完才发现 MATH 无 answer 字段 | 修复 CPU gold extraction；两套数据全量提取单元测试 |
| 继承宿主 torch 版本而非 requirements | 独立 venv；运行前逐一检查全部显式 pin；GPU/runtime 固定 |
| worker 中途失败留下孤儿进程 | Linux parent-death guard + 进程组清理；不做空转 keepalive |
| 重跑覆盖日志/汇总缺条件 | attempt 时间戳、独立逐题原子文件、完整 frozen jobs 验收 |
| 判分异常被记错题 | 超时/异常阻止报告；保留结果，只重跑 CPU |
| shadow 开销被冒充 native 效率 | wall time 标注 instrumented；部署速度需独立硬件基准 |

未包含的更强结论：完整 deficit CDF 的必要性；排除一切 ceiling；所有高维随机方向；跨全部模型的统一机制；“不显著”等于等效；语义解释完全错误。不得凭本任务的某个有利结果宣称这些结论。

小规模验证只验证执行路径和不变量，不能证明 10000-token 全量任务没有任何硬件/数据边界问题。协作者必须在自己的 GPU 上先运行 smoke。正式结果若触发审计失败，停止汇总并保留证据，不降低验收标准。
