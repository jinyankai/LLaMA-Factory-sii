# Qwen3-VL 适配文件总结

## 📦 新增文件清单

为支持 Qwen3-VL，我们创建/更新了以下文件：

### 🔵 核心训练脚本

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `train_qwen3vl_with_layout_loss.py` | Qwen3-VL 专用训练脚本 | ⭐⭐⭐ |
| `test_qwen3vl_compatibility.py` | 兼容性测试套件 | ⭐⭐⭐ |
| `check_qwen3vl_env.py` | 环境检查工具 | ⭐⭐ |

### 📖 文档

| 文件 | 说明 | 适合人群 |
|------|------|----------|
| `QWEN3_VL_UPDATE.md` | 完整适配指南 | 所有人 ⭐⭐⭐ |
| `QWEN3_QUICK_REF.md` | 快速参考卡片 | 快速查阅 ⭐⭐ |
| `QWEN3_FILES_SUMMARY.md` | 本文件（文件清单） | 导航 |

### 🔄 更新的文件

| 文件 | 更新内容 |
|------|----------|
| `README.md` | 添加 Qwen3-VL 章节和快速开始 |
| `layout_aware_loss.py` | 已兼容（无需修改） |

---

## 🚀 快速开始流程

### 步骤 1: 检查环境

```bash
python check_qwen3vl_env.py
```

**输出示例:**
```
==============================================================
  Qwen3-VL 环境检查
==============================================================
✓ Python 版本
  → Python 3.10.12
✓ PyTorch
  → PyTorch 2.1.0 (CUDA 11.8)
✓ Transformers
  → Transformers 4.57.0
✓ Flash Attention
  → Flash Attention 2.5.0
✓ GPU 显存
  → GPU 0: NVIDIA A100 (40.0 GB)
✓ 其他依赖
  → 已安装: accelerate, datasets, tensorboard
✓ 模型类支持
  → AutoModelForImageTextToText 可用

通过: 7/7

✓ 环境已准备好，可以使用 Qwen3-VL！

推荐模型: Qwen3-VL-8B-Instruct（推荐）
```

### 步骤 2: 测试兼容性

```bash
python test_qwen3vl_compatibility.py --model_name Qwen/Qwen3-VL-8B-Instruct
```

**测试内容:**
- ✓ 模型加载
- ✓ Processor 使用
- ✓ 数据预处理
- ✓ 前向传播
- ✓ 布局感知损失计算
- ✓ 生成测试

### 步骤 3: 准备数据

```bash
python omnidoc_to_llamafactory.py \
    --input data/omnidoc_raw.jsonl \
    --output data/omnidoc_processed.json \
    --dataset_name omnidoc_cot \
    --preview
```

### 步骤 4: 开始训练

```bash
python train_qwen3vl_with_layout_loss.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_file data/omnidoc_processed.json \
    --output_dir ./output_qwen3vl \
    --num_epochs 3 \
    --batch_size 2 \
    --use_flash_attn \
    --gradient_checkpointing
```

---

## 📊 文件功能对比

### 训练脚本对比

| 特性 | `train_with_layout_loss.py` | `train_qwen3vl_with_layout_loss.py` |
|------|----------------------------|-------------------------------------|
| 适用模型 | 通用（Qwen-VL, Qwen2-VL 等） | Qwen3-VL 专用 |
| 模型类 | `AutoModelForCausalLM` | `AutoModelForImageTextToText` |
| Flash Attention | 可选 | 内置支持 |
| 数据格式 | ShareGPT | ShareGPT → Qwen3-VL 格式 |
| Processor | 通用 | `AutoProcessor` |
| 推荐使用 | 旧模型 | **Qwen3-VL** ⭐ |

### 测试脚本对比

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| `test_omnibench.py` | 测试数据处理 | 验证数据转换 |
| `test_layout_loss.py` | 测试损失函数 | 验证损失计算 |
| `test_qwen3vl_compatibility.py` | 测试 Qwen3-VL 兼容性 | **训练前必测** ⭐ |
| `check_qwen3vl_env.py` | 检查环境 | 环境诊断 |

---

## 🎯 使用建议

### 新用户（第一次使用）

1. ✅ 阅读 `QWEN3_VL_UPDATE.md`（完整指南）
2. ✅ 运行 `check_qwen3vl_env.py`（检查环境）
3. ✅ 运行 `test_qwen3vl_compatibility.py`（测试兼容性）
4. ✅ 使用 `train_qwen3vl_with_layout_loss.py`（开始训练）

### 快速查阅

- 📖 查看 `QWEN3_QUICK_REF.md`（快速参考）
- 📖 查看 `README.md` 的 Qwen3-VL 章节

### 问题排查

1. 运行 `check_qwen3vl_env.py` 诊断环境
2. 查看 `QWEN3_VL_UPDATE.md` 的"常见问题"章节
3. 运行 `test_qwen3vl_compatibility.py` 定位问题

---

## 🔧 命令行参数速查

### `train_qwen3vl_with_layout_loss.py`

**必需参数:**
```bash
--model_name Qwen/Qwen3-VL-8B-Instruct  # 模型名称
--data_file data.json                    # 数据文件
```

**推荐参数:**
```bash
--output_dir ./output                    # 输出目录
--num_epochs 3                           # 训练轮数
--batch_size 2                           # 批次大小
--use_flash_attn                         # 使用 Flash Attention
--gradient_checkpointing                 # 梯度检查点
```

**损失函数参数:**
```bash
--alpha 1.0                              # LM 损失权重
--beta 0.5                               # BBox 损失权重
--gamma 0.3                              # 关系损失权重
--delta 0.2                              # 顺序损失权重
--bbox_loss_type smooth_l1               # BBox 损失类型
--normalize_coords                       # 归一化坐标
```

**完整示例:**
```bash
python train_qwen3vl_with_layout_loss.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_file data/omnidoc_processed.json \
    --output_dir ./output_qwen3vl \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --use_flash_attn \
    --gradient_checkpointing \
    --alpha 1.0 \
    --beta 0.5 \
    --gamma 0.3 \
    --delta 0.2
```

---

## 📈 性能优化建议

### 显存优化

| 方法 | 节省显存 | 速度影响 | 推荐度 |
|------|----------|----------|--------|
| Flash Attention 2 | ~30% | 加速 | ⭐⭐⭐ |
| Gradient Checkpointing | ~50% | 减速 20% | ⭐⭐⭐ |
| FP8 量化 | ~50% | 轻微减速 | ⭐⭐ |
| 减小批次大小 | 线性 | 减速 | ⭐ |

### 推荐配置

**高性能（A100 40GB）:**
```bash
--model_name Qwen/Qwen3-VL-8B-Instruct \
--batch_size 4 \
--gradient_accumulation_steps 4 \
--use_flash_attn
```

**平衡（RTX 3090 24GB）:**
```bash
--model_name Qwen/Qwen3-VL-8B-Instruct \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--use_flash_attn \
--gradient_checkpointing
```

**节省显存（RTX 3090 24GB）:**
```bash
--model_name Qwen/Qwen3-VL-4B-Instruct \
--batch_size 1 \
--gradient_accumulation_steps 16 \
--use_flash_attn \
--gradient_checkpointing
```

**极限节省（RTX 4090 24GB）:**
```bash
--model_name Qwen/Qwen3-VL-8B-Instruct-FP8 \
--use_fp8 \
--batch_size 1 \
--gradient_accumulation_steps 16 \
--use_flash_attn \
--gradient_checkpointing
```

---

## 🔗 相关链接

### 官方资源
- **Qwen3-VL GitHub**: https://github.com/QwenLM/Qwen3-VL
- **模型下载**: https://huggingface.co/Qwen
- **技术报告**: https://arxiv.org/abs/2505.09388

### 项目文档
- **完整指南**: [`QWEN3_VL_UPDATE.md`](QWEN3_VL_UPDATE.md)
- **快速参考**: [`QWEN3_QUICK_REF.md`](QWEN3_QUICK_REF.md)
- **项目总览**: [`README.md`](README.md)

---

## ✅ 检查清单

### 训练前检查

- [ ] 运行 `check_qwen3vl_env.py` 通过
- [ ] 运行 `test_qwen3vl_compatibility.py` 通过
- [ ] 数据已转换为 ShareGPT 格式
- [ ] 确认 GPU 显存足够
- [ ] 选择合适的模型大小

### 训练中监控

- [ ] 检查 loss 是否下降
- [ ] 监控 GPU 显存使用
- [ ] 查看 TensorBoard 日志
- [ ] 定期保存检查点

### 训练后验证

- [ ] 测试模型生成质量
- [ ] 验证布局理解能力
- [ ] 评估推理速度
- [ ] 保存最终模型

---

## 🎉 开始使用

```bash
# 一键检查环境
python check_qwen3vl_env.py

# 一键测试兼容性
python test_qwen3vl_compatibility.py

# 一键开始训练
python train_qwen3vl_with_layout_loss.py \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --data_file data.json \
    --output_dir ./output
```

**祝训练顺利！🚀**
