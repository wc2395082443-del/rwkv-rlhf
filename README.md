# G1K step1000 独立迁移备份

本分支是独立根提交的备份分支，不合并或改写 main、DAPO-gsm8k、antidoom 等既有分支。
代码、权重、训练/评测原文和数据分别存放，避免把多GB文件塞进Git历史。

## 导航

- [当前训练/推理/奖励代码](code/rl/)
- [SFT代码](code/sft/)
- [启动、评测、数据处理脚本](code/scripts/)
- [源码路径及SHA256对应表](metadata/source_map.json)
- [环境与原始仓库状态](metadata/environment_and_git.json)
- [完整备份Release](https://github.com/wc2395082443-del/rwkv-rlhf/releases/tag/g1k-step1000-backup-20261013-v1)

**状态：已完成上传并发布。19个附件，共10,701,997,123字节，全部通过GitHub返回的SHA256与大小核验。**

- [逐附件下载索引](ASSETS.md)
- [云端校验记录](metadata/upload_receipt.json)
- [训练/评测/数据文件覆盖核验](metadata/coverage_verification.json)：2987份非权重原始文件，missing_count=0；不含17份Jupyter自动保存副本。

本次只上传原始baseline、GSM8K step155和DAPO step1000三个权重，其他checkpoint不在本Release内。
备份完成不代表新服务器环境已验证；建议恢复后确认模型可加载，再释放旧实例。

## 三个模型不要混淆

| 备份标识 | 含义 |
|---|---|
| g1k-baseline.pth | 原始 rwkv-g1k-1b-temp-6003.pth，外部benchmark的原始baseline |
| g1k-gsm8k-step155.pth | GSM8K warmup后step155，是后续DAPO的起点，不是原始baseline |
| g1k-dapo-step1000.pth | 900→1000阶段的final_step_1000.pth |

这些文件不含完整optimizer/scheduler/RNG续训状态，不能据此承诺精确无缝resume。
服务器代码目录名称含g1i，但该共享训练实现也被g1k使用，原路径在source_map中保留。

## Release内容

- g1k-current-code.tar.gz：可浏览代码的原始服务器路径布局。
- server-code-history.tar.gz：历史复现与定制依赖的源码快照；与当前代码分开存放，不引入其他Git分支历史。
- datasets-and-selection.tar.gz：实际保留的训练/验证数据、pass32统计、筛选与交集结果。
- training-originals.tar.gz：服务器现存训练指标、配置、日志和回复，不含其他模型权重。
- evaluation-originals.tar.gz：服务器现存benchmark与教师输出原文、汇总、不同prompt及上下文实验。
- 三个模型的分片：单片最大1GiB；用manifest记录的顺序还原，整文件与分片均提供SHA256。
- backup_manifest.json / file_manifest.json / SHA256SUMS：包、文件及下载资产校验。
- publication_safety.json：公开备份的凭据脱敏、排除项和软链接清单。

本备份只覆盖盘点时服务器实际尚存的内容，不声称恢复之前已清理的文件。
只对公开上传副本进行凭据脱敏，原服务器文件不变；模型和数据不得从这里推定为新的统一授权。
原始模型来自 [BlinkDL/temp-latest-training-models](https://huggingface.co/BlinkDL/temp-latest-training-models)，第三方模型/数据/代码沿用各自上游使用条款，已有许可证文件随源码保存。

## 下载、校验和恢复

需要Python 3.10+。完整提取请在Linux运行，少量历史文件名包含POSIX特殊字符。

```bash
python restore_backup.py --download --assets-dir backup-assets --verify-only
python restore_backup.py --assets-dir backup-assets --output-root restored
```

只恢复三个模型也可使用 `--packages g1k-baseline.pth g1k-gsm8k-step155.pth g1k-dapo-step1000.pth`。
脚本先验证分片和整包SHA256，再恢复相对于服务器根目录的原始路径；默认输出到restored，不覆盖内容不同的已有文件。
将核对后的 restored/hy-tmp 与新机数据路径对齐，再检查tokenizer、reward、prompt、环境和CUDA扩展。

版本列表不是完整容器镜像；缓存、虚拟环境、Git对象历史、私钥和API凭据不随公开备份发布。
软链接只保留清单，不自动覆盖新服务器链接；依赖链接需按实际路径重新建立。
