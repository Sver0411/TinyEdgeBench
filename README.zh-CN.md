# TinyEdgeBench

[English](README.md) | **简体中文**

TinyEdgeBench 在同一个传感器数据任务上对比几种轻量分类方法，看看模型更复杂到底能换来多少准确率、又要为此多付多少存储。

## 它做什么

一个小任务、四种复杂度递增的方法、一份共享数据集：

* 任务：把一小段传感器窗口（最近 8 个采样点）的近期行为分成 `NORMAL`、`RAPID_CHANGE`、`SLOW_DRIFT`、`NOISY`
* 方法：手写规则集、逻辑回归、小决策树、微型 MLP
* 四种方法使用完全相同的 8 个特征、相同的 train / validation / test 行
* 每种方法都给出 accuracy 与 macro F1，用两种不同口径统计体积，并导出成 ESP32 能编译的纯 C

## 运行流程

```text
dataset/generate.py          合成传感器序列 -> 20000 行、8 特征、4 类
training/train.py            训练 logistic / tree / mlp，并对四种方法打分
benchmark/run_benchmark.py   主结果表 + 2 张图，写入 results/
benchmark/measure_flash.py   每次只链接一个 predictor，读取真实 ELF 段大小
training/export_models.py    生成 firmware/main/models/model_*.c（float32 推理）
firmware/                    最小 ESP-IDF 工程，运行被链接进来的 predictor
```

## 模型

| 方法 | 形式 | 设置 |
| --- | --- | --- |
| Rule-based | 基于 `motion` 与 `drift` 的 5 个手写阈值 | 不对数据拟合 |
| Logistic Regression | `StandardScaler` + `LogisticRegression` | max_iter=1000 |
| Decision Tree | `DecisionTreeClassifier` | max_depth=5 |
| Tiny MLP | `StandardScaler` + `MLPClassifier` | 8 → 8 → 8 → 4，ReLU，max_iter=500 |

不引入 TensorFlow、ONNX，也不做量化：v0.1 先把方法之间的比较做清楚。

## 数据集

合成数据。生成器（`dataset/generate.py`，seed 42）先产生短序列的温度 / 湿度 / 光照读数，再摊平成行。

```text
行数       20000（2000 条序列 x 10 行）
特征       8 个 - temp, hum, light, d_temp, d_hum, d_light, motion, drift
类别       4 类 - NORMAL / RAPID_CHANGE / SLOW_DRIFT / NOISY（各 5000 行）
划分       70 / 15 / 15，按序列划分，一条序列不会跨 split
```

* `NORMAL`：基线加传感器噪声
* `RAPID_CHANGE`：1~2 个通道在 3~4 个采样内跳到新电平
* `SLOW_DRIFT`：1~3 个通道持续缓慢单向变化
* `NOISY`：1~3 个通道噪声放大到 3~8 倍，但没有方向性

`motion` 和 `drift` 在 8 个采样的窗口上计算，并做了归一化（1.0 ≈ "和传感器噪声一样大的波动"）。类别之间故意留出重叠：小幅快变看起来像慢漂移，安静但带一点漂移的传感器又像正常。单个特征的准确率最高只到约 61%（`motion`），不到 90%。

> 初始基准使用的是合成数据，真实 ESP32 传感器轨迹留待后续加入。

## 结果

测试集 3000 行，seed 42：

| 方法                | Accuracy | Macro F1 | Raw constants    | Compiled Flash Δ |
| ------------------- | -------: | -------: | :--------------- | ---------------: |
| Rule-based          |   0.6963 |   0.6604 | 20 B             |           968 B  |
| Logistic Regression |   0.8283 |   0.8256 | 208 B            |         1,168 B  |
| Decision Tree       |   0.8203 |   0.8167 | N/A（结构性导出） |         1,308 B  |
| Tiny MLP            |   0.8250 |   0.8214 | 784 B            |         1,904 B  |

怎么读这张表：

* 手写规则明显最弱，但在两种体积口径下也都最便宜
* 逻辑回归在这里既是最准的，也比树和 MLP 更省 flash —— 类边界在 `(motion, drift)` 平面上基本是线性的
* 树和 MLP 与逻辑回归的差异在噪声范围内（0.82 ± 0.01），却要多花 flash
* 决策树没有 "raw constants"：它被导出成嵌套 `if/else` 比较，不存在有意义的参数数组字节数
* 剩余错误主要是 `RAPID_CHANGE` 被判成 `SLOW_DRIFT`：在 8 个采样的窗口里，小幅快变和慢漂移长得很像

### 两个体积列的含义不同

**Raw constants（原始常量）** 是方法理论上要存的数据：权重、偏置、scaler 常量、阈值，按 float32 计数。它描述"模型携带多少信息"，不涉及如何编译，而且四种方法的定义并不一致：

| 方法 | Raw constants | 说明 |
| --- | --- | --- |
| Rule-based | 20 B | 5 个阈值 |
| Logistic Regression | 208 B | 32 权重 + 4 偏置 + 16 scaler |
| Decision Tree | N/A | 无常量 —— 导出为嵌套比较（控制流） |
| Tiny MLP | 784 B | 180 权重/偏置 + 16 scaler |

不要把决策树这一行与其他行并列比较：它根本没有参数数组。因此 `results/benchmark.csv` 里 tree 的 `raw_constants_bytes` 留空，粗略的节点/叶子估算只放在单独的 `structural_estimate_bytes` 列里。

**Compiled Flash Δ（编译后 flash 增量）** 是实测值，不是估算。`benchmark/measure_flash.py` 用 CMake 选项 `-DTINYEDGEBENCH_MODEL=...` 把固件构建五次 —— 一次不含 predictor（baseline），另外每种方法各一次 —— 并从 ESP-IDF 工具链读取 ELF 各段大小：

```text
flash 增量 = 该模型构建的 flash 字节数 - baseline 构建的 flash 字节数

baseline（无 predictor）= 189,492 字节 flash image
```

这是一个编译期 flash 段增量：它包含 predictor 本身，**加上**任何非空构建都会带进来的一小坨固定 demo/脚手架代码（baseline 里这些函数被替换成空实现）。因此它适合在四种方法之间横向比较，但不应被读成纯粹的模型大小。

**比较这四份 C 实现时，请优先看 Compiled Flash Δ** —— 包括 `accuracy_vs_model_size.png`，它的横轴取自 `results/flash_size.csv` 的 compiled flash 增量，而不是 raw constants 计数。

图表：`results/accuracy_vs_model_size.png`（准确率 vs 编译后 flash）、`results/confusion_matrices.png`。原始数据：`results/benchmark.csv`、`results/flash_size.csv`。

## ESP32 状态

| 指标 | 状态 |
| --- | --- |
| 固件构建 | 通过 —— ESP-IDF v5.4.4，`idf.py set-target esp32s3 && idf.py build` |
| 编译后 flash | 构建期实测（无需硬件） |
| 硬件延迟 | 尚未测量 |
| 真实 RAM | 尚未测量 |
| 功耗 | 尚未测量 |
| 单模型 RAM | 未拆分（需要逐模型运行时 profile） |

`firmware/main/main.c` 会把链接进来的 predictor 跑在四行固定数据上并打印结果；`measure_latency()` 是预留的 `esp_timer` 钩子，等板子到了就能用。在此之前不发布任何延迟、RAM 或功耗数字。

按 predictor 统计编译后的 flash（需要有 ESP-IDF 环境：先 source 它的 `export.sh`，或设置 `IDF_PATH`）：

```bash
python benchmark/measure_flash.py     # -> results/flash_size.csv
```

脚本按以下顺序查找 ESP-IDF：`$IDF_PATH/export.sh` → PATH 上已有的 `idf.py`。两者都没有时输出：

```text
ESP-IDF environment not found.
Run ESP-IDF export.sh first or set IDF_PATH.
```

## 运行方式

```bash
python dataset/generate.py          # 生成 dataset/data.csv
python training/train.py            # 训练，打印 val/test 分数
python training/export_models.py    # 生成 firmware/main/models/model_*.c
python benchmark/run_benchmark.py   # 生成 results/benchmark.csv + 2 张图
python benchmark/measure_flash.py   # 生成 results/flash_size.csv（需要 ESP-IDF 环境）
python -m pytest tests/ -v          # 含 Python/C 一致性检查
```

固件（需要 ESP-IDF v5.4+）：

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
```

可复现性：全部固定随机种子，且不保存 pickle —— 导出与基准都用同一个固定 seed 重新训练。

本仓库中的结果由以下版本产生：

```text
Python 3.13.12
NumPy 2.5.3
pandas 3.0.5
scikit-learn 1.9.1
matplotlib 3.11.2
ESP-IDF v5.4.4（target esp32s3）
```

## 项目结构

```text
TinyEdgeBench/
├── dataset/     generate.py, data.csv
├── training/    rules.py（手写规则）, train.py, export_models.py
├── benchmark/   run_benchmark.py, measure_flash.py
├── firmware/    CMakeLists.txt + main/（TINYEDGEBENCH_MODEL 决定链接哪个 predictor）
├── results/     benchmark.csv, confusion.csv, flash_size.csv, 2 张图
├── tests/       test_dataset.py, test_models.py, test_export.py, parity_host.c
├── .github/workflows/tests.yml   checkout + 安装依赖 + pytest
├── README.md / README.zh-CN.md / requirements.txt / LICENSE
```

Python/C 一致性：`tests/parity_host.c` 在主机上编译生成的 C，并用 100 个测试样本与 Python 预测结果逐一比对。

## 局限

* 数据是合成的，因此这些数字还说明不了真实传感器上的表现
* 没有任何硬件实测：无延迟、无功耗、无 RAM profile
* 编译后 flash 增量包含 demo 脚手架，横向可比但不是纯粹的模型体积
* 仅 float32，未做量化
* 一个任务、一种数据规模、一个种子 —— 这是单个数据点，不是普遍规律

## 后续工作

* 真实 ESP32-S3 延迟
* 真实 RAM profiling
* INT8 量化
* 真实传感器轨迹

## 许可

MIT
