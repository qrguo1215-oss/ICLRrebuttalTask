# 表2修正：固定200题的缓存重放实验

此目录是独立任务，复制本目录也能运行，不依赖其他任务目录或原HPC。只补表2，不运行采样校准、八seed采样、GSM或图6。

## 合作者只需执行

需要 Linux、Python 3.10/3.11、支持 BF16 的 NVIDIA GPU。建议每卡至少24GB显存；7B长上下文/重放建议48GB以上。模型自动按冻结版本下载到本目录 `models/`，已有模型可用 `--model-root` 指定目录，加载前检查权重和分词器哈希。

```bash
git clone https://github.com/qrguo1215-oss/ICLRrebuttalTask.git
cd ICLRrebuttalTask/Table2_Revision
bash run.sh plan
bash run.sh run --gpus 0,1,2,3 --workers-per-gpu 1
```

把 `--gpus` 改成自己的可用GPU编号。80GB卡可考虑 `--workers-per-gpu 2`；每个worker仍为单样本，显存不足时恢复为1。默认自动安装固定依赖、运行CPU测试、下载模型、执行GPU安全检查、只生成缺失项、CPU判分、汇总和打包。不需要手工下载向量或拼接历史结果。

完成后只需返回 **`runs/table2_results.tar.gz`**。其中含表格、逐题计数、来源索引、完整新生成、判分结果、安全检查及校验清单。不要只返回汇总截图。

中断后重跑相同命令即可续跑。已完成题目立即落盘；CPU判分失败不会删除或重做GPU结果，可用 `bash run.sh report` 重试。不要删除旧输出以消除报错。

只做预检：`bash run.sh smoke --gpus 0`。预检不生成测试集结果。不同机器可通过 `--configs math15_cps` 等分担完整配置，避免两个控制器写同一输出目录；分机结果合并需按协议哈希与主键核验，不手工覆盖冲突文件。

## 冻结设置

所有六个模型/方法配置使用完全相同的MATH500 `deep200`。这是既有机制诊断子集，不是新抽样的总体准确率评测；题目索引随仓库冻结。每个条件报告整数正确数/200及准确率，不使用120题子集，不做seed均值。

| 模型 | 方法 | 精度 | 干预层 | 运行系数 |
|---|---|---|---|---|
| 1.5B | SEAL | BF16 | 20 | -1 |
| 1.5B | ReBalance | FP16 | 19 | 动态，初始化-1 |
| 1.5B | CPS | BF16 | 20 | +1，向量已含历史缩放 |
| 7B | SEAL | BF16 | 20 | -1 |
| 7B | ReBalance | FP16 | 20 | 动态，初始化-1 |
| 7B | CPS | BF16 | 20 | +1，文件已是-0.6*C5，不再乘-0.6 |

G为无干预greedy；V为原生steering；token replay在同前缀上借用steering选出的下一个token，保留clean分支缓存；thought replay借用完整下一thought文本，再逐token推进clean缓存。干预发生于原生 `<think>` 内split token；完整thought截止于下一split、`</think>`、EOS或全局10,000新token预算，没有128/2048的额外段长上限。下一个split只处理一次。donor缓存和ReBalance动态控制器状态均不泄漏到clean分支。

同尺寸CPS与SEAL共用BF16基线；ReBalance单独保留FP16基线。基线不必跨精度相同，但每行四个条件必须使用同一组200题。

## 最小补跑范围

总共24个表格单元格；共用G后为4,400条独立记录。已随仓库封装2,400条可复用记录：G800、原生V800、replay800。

| 需要生成的条件 | 数量 |
|---|---:|
| 1.5B CPS token/thought replay | 400 |
| 7B SEAL token/thought replay | 400 |
| 7B ReBalance token/thought replay | 400 |
| 7B CPS token/thought replay | 400 |
| 7B ReBalance原生V | 200 |
| 7B CPS原生V | 200 |
| 合计 | **2,000** |

旧7B thought120有2048段长上限且没有正确的think-end停止协议，不能直接补成200题。旧7B ReBalance V跨题状态重置与修订实现不一致，没有足够证据予以豁免，因此仅这200条原生条件重做。新7B CPS向量可用，但其旧评测未明确记录精度，不能直接当作BF16参考。

## 证据边界和失败处理

- 已导入的原始生成保留原run_id、运行环境和文件哈希，不伪装成当前代码新生成。原部分包缺少顶层RUN.json；执行前自动用两道校准题完整greedy序列核验历史基线，并做恒等/自重放/缓存与动态状态隔离检查。失败即停，不自动重跑所有参考。
- 7B SEAL原生200题的原始响应已回收，逐文件哈希与独立审计记录一致。所有复用及新生成均使用同一CPU评测器重新判分，历史标签只作差异诊断。旧SEAL7/CPS15文本已经裁剪，无法追溯未裁剪输出，因此其严格评分敏感性留空，不合成数值。
- 7B CPS历史系数曾在MATH500上选择；本任务复现其机制对照，不把它写成新的held-out系数验证。
- 算法、数据、向量、权重、精度、停止规则不能为省时调整。调度采用按基线长度排序的动态队列；每条任务随机状态独立、batch为1。高GPU利用率不代替有效吞吐，本任务不附带占用GPU的keepalive。
- 完整表格只有每格200条、全部判分成功后才输出。超时/解析错误不自动计错；缺失不缩分母；新旧评分差异保存在 `historical_grading_changes.json`，主表始终采用同一评测器结果，不混用旧标签。

`assets/reuse_scope.json`给出逐条待运行任务，`assets/checksums.json`给出输入哈希。`summary/table2.csv`包含正确数、200分母、准确率、相对本行G的收益和复用来源计数；`summary/per_item.csv`支持逐题核对。
