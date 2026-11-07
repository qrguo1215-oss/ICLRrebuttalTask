# 发布前验证记录

本记录描述软件验证，不是正式补跑结果，也不报告模型准确率。

## 已实际执行

- 15个自动化测试通过，包括资产校验、100/400与deep200划分、任务去重、完整thought结束条件、低精度deficit坐标、模型覆盖文件拒绝、判分敏感性、方法独立汇总、CPU判分/报告/打包，以及控制进程被SIGKILL后worker退出。
- 两张A800上完成以下7个配置的短程检查：MATH 1.5B SEAL/CPS/ReBalance、MATH 7B SEAL/ReBalance、GSM 1.5B ReBalance、GSM 7B SEAL。
- 每个配置仅使用一个校准题，每次生成最多256个token；没有用heldout题调试。全部通过关闭/零干预、自身token/thought replay、同seed重复、影子前向不变性及KV/控制器隔离检查。ReBalance额外通过跨题历史污染测试。
- 上述7个配置都走通了“基线生成 → 保存token IDs → frozen-prefix scan → W1校准 → S/S+V和E/E+V生成”的实际流程。
- 1.5B/7B的本地完整权重经过SHA256计算，并与固定Hub提交的LFS哈希核对；配置、tokenizer、分片索引也与下载文件逐一核对。没有重新下载整套大权重；真实GPU检查使用内容相同的已有权重。

实测环境为Python 3.10、PyTorch 2.3.1/CUDA 12.1、Transformers 4.46.3、NumPy 1.26.4。tokenizers 0.20.3及safetensors 0.8.0已固定到实测版本。CUDA cumsum的warn-only例外在运行元数据中明确记录。

## 验证边界

没有启动90000条正式生成，没有据此声称论文结论已复现。256-token smoke不能代替长上下文资源测试；完整thought超过2048词元的停止逻辑另由单元测试覆盖。不同机器仍须运行自动预检，不能复用本次PASS记录绕过检查。

小样本产物不随Git提交，也不会被正式汇总接受。最终结果仍须由合作者运行完整协议后产生。
