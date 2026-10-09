<h1 align="center">TinyEdgeBench</h1>

<p align="center"><strong>ESP32-S3 轻量分类器与 FP32 / INT8 权衡</strong></p>

<p align="center">共享数据 · 量化比较 · 资源实测</p>

---

<p align="center">
  <a href="#project-overview">概览</a> ·
  <a href="#研究问题与方法">方法</a> ·
  <a href="#实现与测量状态">验证状态</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">主项目关联</a> ·
  <a href="README.md">English</a>
</p>

---

<a id="project-overview"></a>

TinyEdgeBench 在共享数据的传感器窗口任务上比较 Rule、Logistic、Tree 和 MLP，并在真实 ESP32-S3 上测量资源与延迟。v0.3 研究 Logistic 和 MLP 训练后 INT8 量化的 Flash、速度与准确率权衡；Rule 和 Tree 保持原生实现作为控制组。

## 研究问题与方法

更小的数值表示能否改善实际 MCU 推理？任务将包含 8 个传感器读数的窗口分为 `NORMAL`、`RAPID_CHANGE`、`SLOW_DRIFT` 或 `NOISY`。共享的 8 维特征为温度、湿度、光照、三个最新变化量、`motion` 和 `drift`。

| 模型 | 实现 |
| --- | --- |
| Rule | 对 motion 和 drift 使用 5 个固定阈值 |
| Logistic | StandardScaler + 多类别逻辑回归 |
| Tree | 最大深度 5 的决策树，导出为分支代码 |
| MLP | StandardScaler + 8→8→8→4 ReLU 网络 |

数据集有 20,000 条合成数据（seed 42），同一序列不跨划分。准确率和 Macro F1 来自完整的 3,000 条测试样本；共享的 256 条板端子集用于 C/Python 预测一致性检查。尚未测量真实农业数据准确率。

## 主要结果：v0.3 FP32 / INT8

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

<!-- V03_COMPARE_START -->
| Pair | Δ accuracy | Δ Macro F1 | Flash saving | Raw p50 change |
| --- | ---: | ---: | ---: | ---: |
| Logistic INT8 − FP32 | +0.0000 | -0.0004 | +32 B (+4.9%) | +2.078 µs (+90.5%) |
| Mlp INT8 − FP32 | -0.0030 | -0.0036 | +292 B (+20.9%) | +5.391 µs (+64.5%) |
<!-- V03_COMPARE_END -->

**当前便携式标量 INT8 实现节省 Flash，但推理更慢。** Logistic 保持完整测试集准确率，Macro F1 略降；MLP 的准确率和 F1 均下降。Tree 的原始批次平均延迟最低，FP32 Logistic 的准确率和 Macro F1 最高。这些结论仅描述当前开发板、任务与编译配置，不能推广至所有 INT8 运行时。

Flash Δ 是相对 v0.3 无模型框架的编译 Flash 段差值，原始参数是导出常数。本次构建六种实现的 Static RAM Δ 和推理堆分配量均为 0 B，不等于不占 RAM。详见[完整硬件结果](results/v0.3/hardware_benchmark.csv)、[v0.3 渲染图](results/v0.3/plots/)和[整数核心证据](results/v0.3/integer_core_evidence.txt)。

## 实现与测量状态

- Logistic 和 MLP 均有 FP32、INT8 C 实现。校准只使用训练集；INT8 权重与激活使用 int32 累加、隐藏层整数定点重标定。输入 StandardScaler 预处理仍使用 FP32，未评估优化的 ESP-NN/TFLM 内核。
- 真机计时使用 ESP32-S3 160 MHz、ESP-IDF v5.4.4、performance 优化，关闭 Wi-Fi/蓝牙。每种实现运行三次，计时前核对 256/256 条预测，再预热 100 次，执行 2,000 批、每批 64 次推理。
- 表中是未扣 baseline 的单次推理批次平均原始延迟，包含预处理与计时器开销。百分位数属于批次平均值，不是单次调用尾延迟；成对 timing baseline 与扣除值保存在 CSV 中。
- Flash/静态 RAM 来自编译，延迟/堆来自真机，准确率/F1 来自合成数据完整测试集。功耗、能耗及有标签的真实传感器准确率尚未测量。

精确定义见[量化方法](docs/quantization_methodology.md)、[测量方法](docs/methodology.md)和[硬件流程](docs/hardware_protocol.md)。

## 快速开始与复现

在仓库根目录准备主机环境并导出模型：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python training/export_models.py
python training/quantize.py
python benchmark/export_dataset.py
python benchmark/run_benchmark.py
```

重复 v0.3 真机测量需要 ESP-IDF v5.4+ 环境、已连接的 ESP32-S3 和干净的代码提交：

```bash
python benchmark/v03_run_hardware.py --port /dev/cu.usbmodemYOURBOARD --runs 3
python benchmark/plot_v03.py
python benchmark/update_readme_v03.py
```

正式结果写入 `results/v0.3/`；`--allow-dirty` 仅用于探索测量。完整流程见[量化指南](docs/quantization_methodology.md)。`benchmark/update_readme.py` 只更新归档的 v0.2/离线区域；`benchmark/update_readme_v03.py` 更新当前比较。

## Relationship to Smart-Agriculture-Edge-AI

[Smart-Agriculture-Edge-AI](https://github.com/Sver0411/Smart-Agriculture-Edge-AI) 将 TinyEdgeBench 作为轻量模型与资源评估参考。这些独立计算基准与主系统固件不同，不代表主系统的板端性能。

## 局限与后续工作

当前仅覆盖一个合成数据集和 seed、一种板端配置，以及标量 FP32/INT8 实现。真实农业数据、功耗/能耗测量及优化整数内核属于后续方向。微秒级计时受到计时开销和中断影响；子集准确率与完整测试集准确率分开记录。

## v0.2 历史证据

v0.2 FP32 结果保留于 [`v0.2` 标签](https://github.com/Sver0411/TinyEdgeBench/tree/v0.2)及下方。**v0.2 和 v0.3 使用不同的资源测量框架**，不能把跨版本 Flash 差值归因于量化，应使用上方 v0.3 的成对比较。

<details>
<summary>展开 v0.2 FP32 真机与离线结果表</summary>

<!-- HARDWARE_TABLE_START -->
| Model | Accuracy (3,000) | Macro F1 (3,000) | Flash Δ | Static RAM Δ | Inference heap alloc | p50 µs | p95 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 40 B | 0 B | 0 B | 0.7 ± 0.0 | 0.7 ± 0.0 | 0.7 ± 0.0 |
| Logistic | 0.8283 | 0.8256 | 204 B | 0 B | 0 B | 2.3 ± 0.0 | 2.3 ± 0.0 | 2.4 ± 0.0 |
| Tree | 0.8203 | 0.8167 | 284 B | 0 B | 0 B | 0.3 ± 0.0 | 0.4 ± 0.0 | 0.4 ± 0.0 |
| MLP | 0.8250 | 0.8214 | 944 B | 0 B | 0 B | 8.4 ± 0.0 | 8.5 ± 0.0 | 8.5 ± 0.0 |
<!-- HARDWARE_TABLE_END -->

准确率/F1 来自 3,000 条合成测试样本，Flash/静态 RAM 来自编译，延迟/堆来自真机。256 条子集的准确率另见[硬件证据](results/hardware/)。

<!-- OFFLINE_TABLE_START -->
| Model | Accuracy | Macro F1 | Raw constants |
| --- | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 20 B |
| Logistic | 0.8283 | 0.8256 | 208 B |
| Tree | 0.8203 | 0.8167 | N/A (control flow) |
| MLP | 0.8250 | 0.8214 | 784 B |
<!-- OFFLINE_TABLE_END -->

原始常数统计 float32 阈值、权重、偏置和标准化参数；导出树属于控制流。[离线结果](results/benchmark.csv)、[混淆计数](results/confusion.csv)、[准确率/体积图](results/accuracy_vs_model_size.png)和[混淆矩阵图](results/confusion_matrices.png)均保留。

</details>

## 许可证

MIT — 见 [LICENSE](LICENSE)。
