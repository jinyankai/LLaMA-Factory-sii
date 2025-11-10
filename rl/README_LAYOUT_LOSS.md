# 布局感知损失函数 - 完整方案

## 🎯 问题与解决方案

### 问题
标准的语言模型训练只优化文本生成，对于文档理解任务：
- ❌ 模型可能生成"看起来对"但坐标错误的文本
- ❌ 关系识别不准确（parent_son, truncated）
- ❌ 阅读顺序理解不足

### 解决方案
**多任务损失函数**，同时优化：
- ✅ 文本生成质量（语言建模）
- ✅ 坐标预测准确性（边界框回归）
- ✅ 关系识别能力（关系分类）
- ✅ 顺序理解能力（顺序预测）

---

## 📦 文件清单

| 文件 | 说明 | 用途 |
|------|------|------|
| `layout_aware_loss.py` | 核心损失函数实现 | ⭐ 核心 |
| `train_with_layout_loss.py` | 完整训练脚本 | 🚀 训练 |
| `test_layout_loss.py` | 测试套件 | 🧪 测试 |
| `CUSTOM_LOSS_GUIDE.md` | 详细使用指南 | 📖 文档 |
| `README_LAYOUT_LOSS.md` | 本文件 | 📋 概览 |

---

## ⚡ 快速开始

### 1️⃣ 测试损失函数

```bash
python test_layout_loss.py
```

**预期输出：**
```
============================================================
布局感知损失函数测试套件
============================================================

测试 1: 布局信息提取
✅ 信息提取测试通过！

测试 2: 边界框损失
✅ 边界框损失测试通过！

...

🎉 所有测试通过！
```

### 2️⃣ 训练模型

```bash
python train_with_layout_loss.py \
    --model_name Qwen/Qwen-VL-Chat \
    --data_file data/omnidoc_processed.json \
    --output_dir ./output \
    --num_epochs 3 \
    --batch_size 2 \
    --alpha 1.0 \
    --beta 0.5 \
    --gamma 0.3 \
    --delta 0.2
```

### 3️⃣ 监控训练

```bash
tensorboard --logdir ./output/logs
```

---

## 🔬 损失函数详解

### 数学公式

```
L_total = α·L_lm + β·L_bbox + γ·L_relation + δ·L_order
```

### 各项损失

#### 1. 语言建模损失 (L_lm)

```python
L_lm = CrossEntropy(logits, labels)
```

- **作用**：标准的文本生成损失
- **权重**：α = 1.0（基准）

#### 2. 边界框损失 (L_bbox)

```python
# Smooth L1 Loss
L_bbox = SmoothL1(pred_bbox, target_bbox)

# 或 IoU Loss
L_bbox = 1 - IoU(pred_bbox, target_bbox)
```

- **作用**：优化坐标预测
- **权重**：β = 0.5
- **输入**：从文本提取 `[x1, y1, x2, y2]`

#### 3. 关系分类损失 (L_relation)

```python
L_relation = CrossEntropy(pred_relation, target_relation)
```

- **作用**：优化关系识别
- **权重**：γ = 0.3
- **类别**：none, parent_son, truncated, sibling

#### 4. 阅读顺序损失 (L_order)

```python
L_order = L1(pred_order, target_order)
```

- **作用**：优化顺序预测
- **权重**：δ = 0.2

---

## 📊 实验结果（预期）

### 消融实验

| 配置 | Perplexity ↓ | Bbox IoU ↑ | Relation Acc ↑ | Order MAE ↓ |
|------|--------------|------------|----------------|-------------|
| Baseline (α=1.0) | 3.45 | 0.65 | 0.72 | 2.3 |
| +Bbox (β=0.5) | 3.42 | **0.78** | 0.73 | 2.2 |
| +Relation (γ=0.3) | 3.40 | 0.79 | **0.85** | 2.1 |
| Full (δ=0.2) | **3.38** | 0.80 | 0.86 | **1.8** |

### 权重调优

| α | β | γ | δ | 总体性能 | 适用场景 |
|---|---|---|---|----------|----------|
| 1.0 | 0.0 | 0.0 | 0.0 | 基准 | 纯文本生成 |
| 1.0 | 1.0 | 0.2 | 0.2 | ⭐⭐⭐ | 坐标密集任务 |
| 1.0 | 0.3 | 0.8 | 0.2 | ⭐⭐⭐⭐ | 关系密集任务 |
| 1.0 | 0.5 | 0.5 | 0.5 | ⭐⭐⭐⭐⭐ | 平衡（推荐） |

---

## 🎨 使用示例

### 示例 1：基础使用

```python
from layout_aware_loss import LayoutAwareLoss

# 创建损失函数
loss_fn = LayoutAwareLoss(
    alpha=1.0,
    beta=0.5,
    gamma=0.3,
    delta=0.2
)

# 计算损失
loss_dict = loss_fn(
    logits=model_output.logits,
    labels=batch["labels"],
    pred_texts=decoded_predictions,
    target_texts=decoded_targets
)

# 反向传播
loss_dict["loss"].backward()
```

### 示例 2：集成到 Trainer

```python
from layout_aware_loss import create_layout_aware_trainer

trainer = create_layout_aware_trainer(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    train_dataset=train_dataset,
    loss_config={
        "alpha": 1.0,
        "beta": 0.5,
        "gamma": 0.3,
        "delta": 0.2,
    }
)

trainer.train()
```

### 示例 3：自定义权重

```python
# 场景：主要优化坐标准确性
loss_fn = LayoutAwareLoss(
    alpha=1.0,
    beta=1.0,   # 增加边界框权重
    gamma=0.2,
    delta=0.2,
    bbox_loss_type="iou"  # 使用 IoU Loss
)
```

---

## 🔧 高级配置

### 1. 动态权重调整

```python
class DynamicWeightLoss(LayoutAwareLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
    
    def forward(self, *args, **kwargs):
        # 前期关注语言建模，后期关注布局
        self.beta = min(0.5, self.step / 10000 * 0.5)
        self.step += 1
        return super().forward(*args, **kwargs)
```

### 2. 选择性计算

```python
# 只在验证时计算布局损失（节省训练时间）
class SelectiveLoss(LayoutAwareLoss):
    def forward(self, logits, labels, pred_texts=None, target_texts=None, training=True):
        if training:
            # 训练时不解码文本
            pred_texts = None
            target_texts = None
        return super().forward(logits, labels, pred_texts, target_texts)
```

### 3. 加权采样

```python
# 对包含更多布局信息的样本增加权重
from torch.utils.data import WeightedRandomSampler

def compute_sample_weight(sample):
    text = sample["conversations"][-1]["value"]
    # 统计布局信息数量
    num_bboxes = len(extractor.extract_bboxes(text))
    num_relations = len(extractor.extract_relations(text))
    return 1.0 + 0.5 * (num_bboxes + num_relations)

weights = [compute_sample_weight(s) for s in dataset]
sampler = WeightedRandomSampler(weights, len(weights))
```

---

## 📈 性能优化

### 1. 减少解码开销

布局损失需要解码文本，这会增加 20-30% 的训练时间。

**优化方案：**

```python
# 方案 A: 只在验证时计算
trainer.decode_for_layout_loss = False  # 训练时
trainer.decode_for_layout_loss = True   # 验证时

# 方案 B: 每 N 步计算一次
if step % 10 == 0:
    trainer.decode_for_layout_loss = True
else:
    trainer.decode_for_layout_loss = False

# 方案 C: 使用缓存
# 缓存最近的解码结果，避免重复解码
```

### 2. 混合精度训练

```python
training_args = TrainingArguments(
    fp16=True,  # 或 bf16=True
    gradient_checkpointing=True,
)
```

### 3. 梯度累积

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

---

## 🐛 故障排除

### 问题 1: 损失值异常大

**原因**：坐标未归一化

**解决**：
```python
loss_fn = LayoutAwareLoss(
    normalize_coords=True,
    page_size=(1200, 1684)
)
```

### 问题 2: 布局损失始终为 0

**原因**：文本中没有布局信息

**检查**：
```python
from layout_aware_loss import LayoutInfoExtractor

extractor = LayoutInfoExtractor()
text = "your text here"
print("Bboxes:", extractor.extract_bboxes(text))
print("Relations:", extractor.extract_relations(text))
```

### 问题 3: 训练不稳定

**解决**：
1. 降低布局损失权重
2. 使用梯度裁剪
3. 减小学习率

```python
training_args = TrainingArguments(
    max_grad_norm=1.0,  # 梯度裁剪
    learning_rate=1e-5,  # 降低学习率
)

loss_fn = LayoutAwareLoss(
    beta=0.2,  # 降低权重
    gamma=0.1,
    delta=0.1,
)
```

### 问题 4: OOM（显存不足）

**解决**：
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
)

# 关闭训练时的文本解码
trainer.decode_for_layout_loss = False
```

---

## 📚 扩展阅读

### 相关论文

1. **Multi-task Learning**
   - [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/abs/1706.05098)

2. **Document Understanding**
   - [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
   - [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)

3. **Loss Function Design**
   - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
   - [GIoU: Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)

### 相关项目

- **LLaMAFactory**: https://github.com/hiyouga/LLaMA-Factory
- **Qwen-VL**: https://github.com/QwenLM/Qwen-VL
- **OmniDocBench**: https://github.com/opendatalab/OmniDocBench

---

## 🎓 最佳实践

### ✅ 推荐做法

1. **先测试再训练**
   ```bash
   python test_layout_loss.py
   ```

2. **从小权重开始**
   ```python
   loss_config = {"alpha": 1.0, "beta": 0.1, "gamma": 0.1, "delta": 0.1}
   ```

3. **监控各项损失**
   - 使用 TensorBoard 或 wandb
   - 确保所有损失都在下降

4. **消融实验**
   - 逐步添加损失项
   - 对比效果提升

5. **数据质量检查**
   - 确保文本包含布局信息
   - 验证格式正确

### ❌ 避免的做法

1. ❌ 不测试直接训练
2. ❌ 权重设置过大（β, γ, δ > 1.0）
3. ❌ 忽略损失监控
4. ❌ 数据格式不统一
5. ❌ 不做消融实验

---

## 🚀 下一步

1. ✅ 运行测试：`python test_layout_loss.py`
2. ✅ 准备数据：确保包含布局信息
3. ✅ 小规模训练：验证流程
4. ✅ 调整权重：根据任务优化
5. ✅ 全量训练：获得最终模型
6. ✅ 评估效果：对比 baseline

---

## 💬 反馈与支持

遇到问题？
- 查看 `CUSTOM_LOSS_GUIDE.md` 详细文档
- 运行 `test_layout_loss.py` 诊断
- 检查数据格式是否正确

**祝训练顺利！🎉**
