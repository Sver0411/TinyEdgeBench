# TinyEdgeBench v0.3

[English](README.md) | **简体中文**

TinyEdgeBench 在同一传感器窗口任务、同一组 8 维特征、同一测试集和同一块 ESP32-S3 上比较四种轻量分类器。v0.3 研究原有 Logistic 与 MLP 经过训练后 INT8 量化的实际变化；Rule 和 Tree 继续作为控制组。已发布的 v0.2 证据保留在下方及 [`v0.2` 标签](https://github.com/Sver0411/TinyEdgeBench/tree/v0.2)。

## v0.3：FP32 与 INT8

便携式标量 INT8 使用仅来自训练集的校准、INT8 权重和激活、int32 累加，以及隐藏层整数定点重标定。输入 StandardScaler 预处理仍使用 FP32。六种实现采用相同数据、ESP32-S3 和批次计时协议。主表延迟是**未扣除 baseline 的原始值**。

<!-- V03_MAIN_START -->
| Model | Precision | Test accuracy | Macro F1 | Flash Δ | Raw params | Raw p50 µs | Raw p95 µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | NATIVE | 0.6963 | 0.6604 | 492 B | 20 B | 0.672 ± 0.000 | 0.688 ± 0.000 |
| Tree | NATIVE | 0.8203 | 0.8167 | 752 B | N/A | 0.344 ± 0.000 | 0.359 ± 0.000 |
| Logistic | FP32 | 0.8283 | 0.8256 | 656 B | 208 B | 2.297 ± 0.000 | 2.297 ± 0.000 |
| Logistic | INT8 | 0.8283 | 0.8253 | 624 B | 128 B | 4.375 ± 0.000 | 4.391 ± 0.000 |
| MLP | FP32 | 0.8250 | 0.8214 | 1,396 B | 784 B | 8.359 ± 0.000 | 8.484 ± 0.000 |
| MLP | INT8 | 0.8220 | 0.8178 | 1,104 B | 354 B | 13.750 ± 0.000 | 13.875 ± 0.000 |
<!-- V03_MAIN_END -->

### 量化效果

<!-- V03_COMPARE_START -->
| Pair | Δ accuracy | Δ Macro F1 | Flash saving | Raw p50 change |
| --- | ---: | ---: | ---: | ---: |
| Logistic INT8 − FP32 | +0.0000 | -0.0004 | +32 B (+4.9%) | +2.078 µs (+90.5%) |
| Mlp INT8 − FP32 | -0.0030 | -0.0036 | +292 B (+20.9%) | +5.391 µs (+64.5%) |
<!-- V03_COMPARE_END -->

[v0.3 量化方法](docs/quantization_methodology.md)定义了校准、舍入、饱和、整数累加、存储计数及成对 timing baseline 扣除。每类 F1 和混淆矩阵保存在 `results/v0.3/`。功耗与真实传感器准确率：**Not measured yet**。

在这项任务与 ESP32-S3 配置下，两种标量 INT8 实现均节省了编译 Flash，但原始推理延迟增加。Logistic 保持完整测试集准确率，Macro F1 略降；MLP 的准确率和 Macro F1 都有所下降。本次构建六种实现的 Static RAM Δ 与推理堆分配量均为 0 B，这不代表模型不占 RAM。成对 timing baseline 与扣除后的数值保存在[硬件结果 CSV](results/v0.3/hardware_benchmark.csv)。另见[四张 v0.3 图](results/v0.3/plots/)及[整数核心反汇编检查](results/v0.3/integer_core_evidence.txt)。

## 已归档的 v0.2 FP32 基线

<!-- HARDWARE_TABLE_START -->
| Model | Accuracy (3,000) | Macro F1 (3,000) | Flash Δ | Static RAM Δ | Inference heap alloc | p50 µs | p95 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 40 B | 0 B | 0 B | 0.7 ± 0.0 | 0.7 ± 0.0 | 0.7 ± 0.0 |
| Logistic | 0.8283 | 0.8256 | 204 B | 0 B | 0 B | 2.3 ± 0.0 | 2.3 ± 0.0 | 2.4 ± 0.0 |
| Tree | 0.8203 | 0.8167 | 284 B | 0 B | 0 B | 0.3 ± 0.0 | 0.4 ± 0.0 | 0.4 ± 0.0 |
| MLP | 0.8250 | 0.8214 | 944 B | 0 B | 0 B | 8.4 ± 0.0 | 8.5 ± 0.0 | 8.5 ± 0.0 |
<!-- HARDWARE_TABLE_END -->

上表的准确率与 F1 来自**完整 3,000 条合成测试样本**；Flash 和静态 RAM 来自编译；延迟和堆来自真实 ESP32-S3。256 条真机子集用于预测一致性检查，其准确率另见[真机结果](results/hardware/)；表格由 `benchmark/update_readme.py` 生成。功耗：**Not measured yet**。

推理堆分配量 0 B 表示这些推理函数没有动态分配，并不表示模型不占用 RAM。

## 真机验证

ESP32-S3 以 160 MHz、ESP-IDF v5.4.4 和 performance 优化运行，Wi-Fi 与蓝牙均未启用。四个模型各重复三次；每次计时前均逐条核对 **256/256 条预测与 Python 参考结果一致**，然后预热 100 次，并运行 2,000 组、每组 64 次推理。256 条子集用于实现一致性检查；上方质量指标来自完整测试集。

在这组合成任务与开发板配置下，Logistic 的完整测试集准确率和 Macro F1 均为四种方法中最高，新增编译 Flash 为 204 B，而本实验 MLP 为 944 B。Tree 的批次平均延迟最低。这些比较只描述当前任务和配置。

## 任务与模型

将最近 8 个传感器读数分为 `NORMAL`、`RAPID_CHANGE`、`SLOW_DRIFT`、`NOISY`。四种方法都使用温度、湿度、光照、三个最新变化量、`motion`、`drift`：

| 模型 | 实现 |
| --- | --- |
| Rule | 对 motion 和 drift 使用 5 个固定阈值 |
| Logistic | StandardScaler + 多类别逻辑回归 |
| Tree | 最大深度 5 的决策树，导出为分支代码 |
| MLP | StandardScaler + 8→8→8→4 ReLU 网络 |

`dataset/generate.py` 使用种子 42 生成 20,000 条**合成**数据，同一序列不会跨越训练、验证和测试划分。完整测试集有 3,000 条。`benchmark/export_dataset.py` 从每类 64 条不同测试序列各取一个窗口并交错排列，导出 256 条 float32 特征、真实标签和 Python 预测参考值。四个真机构建使用完全相同的样本；不会为个别模型修改数据。

## 离线结果：完整测试集

<!-- OFFLINE_TABLE_START -->
| Model | Accuracy | Macro F1 | Raw constants |
| --- | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 20 B |
| Logistic | 0.8283 | 0.8256 | 208 B |
| Tree | 0.8203 | 0.8167 | N/A (control flow) |
| MLP | 0.8250 | 0.8214 | 784 B |
<!-- OFFLINE_TABLE_END -->

这些 Accuracy 和 Macro F1 来自 Python 在 3,000 条合成测试样本上的评价。原始常数只计 float32 阈值、权重、偏置和标准化参数；导出为控制流的树没有可比较的参数数组。[离线 CSV](results/benchmark.csv)与[混淆矩阵 CSV](results/confusion.csv)保存完整数据。

## 指标定义

- **编译：** `benchmark/measure_flash.py` 对无模型 baseline 和四个模型分别编译，保留同一数据集与测量框架。Flash Δ 为 ELF 中写入 Flash 的段总量减 baseline；静态 RAM Δ 为 `.data + .bss` 减 baseline。它们包含少量模型选择代码差异，不能当成纯参数体积。[编译结果 CSV](results/flash_size.csv)。
- **真机正确性：** 计时前逐条核对 256 个 C 预测与 Python 参考值；不一致则中止。设备还输出标签正确数、混淆矩阵和 checksum。
- **真机延迟：** 每个计时批次连续调用 C predictor 64 次，再将批次时间除以 64；共 2,000 批次、预热 100 次，轮流使用全部 256 条样本。中断保持开启。批次内没有串口、分配或日志。报告带小数精度的单次平均 min、mean、p50、p95、p99、max；百分位数基于批次平均值，计时器开销未扣除。
- **运行期 RAM：** 分开报告推理前后 8-bit 可用堆及自启动以来的最小可用堆。主表的推理堆分配量是前后差值，不是自启动以来的峰值。主任务栈 high-water mark 仅保留在机器可读结果中作为任务级诊断，不作为模型指标。
- **环境：** 每次仅链接一个模型。固件未初始化 Wi-Fi 或蓝牙。设备自动输出 CPU 频率、ESP-IDF 版本、优化级别、日志级别、Git 提交和脏状态、源码哈希、模型哈希、配置哈希、数据集哈希和运行设置。解析器拒绝不完整日志、数据集不符、编译尺寸与固件来源不符的结果。这是独立计算 benchmark，不代表完整应用延迟。

详见[实验方法](docs/methodology.md)和[硬件操作说明](docs/hardware_protocol.md)。

## 复现

使用 ESP-IDF v5.4+ 环境及 `requirements.txt` 中的 Python 依赖：

```bash
python -m pytest tests/ -q
python training/export_models.py
python benchmark/export_dataset.py
python benchmark/run_benchmark.py
python benchmark/run_hardware.py --port /dev/cu.usbmodemYOURBOARD --runs 3
python benchmark/plot_hardware.py
python benchmark/update_readme.py
```

先提交代码并确保工作区干净，再运行 `run_hardware.py`。它测量 Flash、编译、刷机、重启、读取串口、校验记录，并写入 `results/flash_size.csv`、`results/hardware/hardware_benchmark.csv`、`hardware_metadata.json` 和已清理设备标识的 `raw_serial_logs/`。`--allow-dirty` 只用于探索性测量，不能进入 README 表格。已有日志可用 `python benchmark/parse_hardware_log.py serial.log --build-sizes results/flash_size.csv` 解析。手动单模型构建可使用 `idf.py -DTINYEDGEBENCH_MODEL=rule build`，或改为 `logistic`、`tree`、`mlp`。`baseline` 仅用于资源差值计算。

## 局限

- 标签与准确率来自合成数据；真实传感器轨迹和有可靠标签的真实环境准确率：**Not measured yet**。
- 功耗、能耗：**Not measured yet**；目前未发现可靠电流测量仪器。
- 仅 FP32、单数据集、单 seed、单板配置。INT8 留到 FP32 基准稳定之后。
- 256 条真机子集与 3,000 条完整测试集不同，因此两个准确率不一定相同。
- 批次计时提高了分辨率，但计时开销和中断活动仍会影响相近模型的对比；批次百分位数不等于单次调用尾延迟。

保留原有[离线准确率与 Flash 图](results/accuracy_vs_model_size.png)及[混淆矩阵图](results/confusion_matrices.png)。硬件图仅根据 `results/hardware/` 中已校验的结果生成。

## 许可证

MIT，见 [LICENSE](LICENSE)。
