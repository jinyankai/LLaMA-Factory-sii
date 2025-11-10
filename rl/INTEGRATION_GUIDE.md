# LLaMAFactory 自定义数据处理集成指南

## 问题场景

你想在 LLaMAFactory 中使用 OmniDocBench 数据集，并且需要：
1. 从 layout 标注生成思维链
2. 在训练时动态处理数据
3. 保持 LLaMAFactory 的训练流程

## 解决方案

### 方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **A. 预处理** | 简单、无需改 LLaMAFactory | 数据固定、占用存储 | 数据量小、处理逻辑稳定 |
| **B. 运行时处理** | 灵活、节省存储 | 需要修改 LLaMAFactory | 数据量大、需要动态处理 |

---

## 方案 A：预处理方式（推荐新手）

### 步骤 1：生成处理后的数据

```bash
# 使用你的 omnibench.py 生成数据
python omnibench.py --input raw_data.jsonl --output processed_data.json
```

### 步骤 2：创建 dataset_info.json

```json
{
  "omnidoc_cot": {
    "file_name": "processed_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

### 步骤 3：训练

```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen-VL-Chat \
  --dataset omnidoc_cot \
  --dataset_dir ./data \
  --output_dir ./output
```

---

## 方案 B：运行时处理（推荐进阶）

### 步骤 1：修改 LLaMAFactory 源码

在 `LLaMAFactory/src/llamafactory/data/loader.py` 中添加：

```python

# 在文件开头导入
from typing import TYPE_CHECKING, Literal, Union
import os
import json

# 添加自定义加载函数
def load_omnidocbench_dataset(
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    """加载 OmniDocBench 并应用思维链转换"""
    from datasets import Dataset
    
    # 导入你的转换器（需要将 omnibench.py 放到可导入的位置）
    import sys
    sys.path.append("/path/to/your/dataset/folder")
    from omnibench import OmniDocConverter
    
    # 读取原始数据
    dataset_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    raw_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    # 应用转换
    converter = OmniDocConverter()
    processed_data = []
    for page_data in raw_data:
        qa_pairs = converter.process_page(page_data)
        processed_data.extend(qa_pairs)
    
    # 转换为 ShareGPT 格式
    formatted_data = []
    for item in processed_data:
        conversations = []
        images = []
        
        for msg in item["messages"]:
            role_map = {"system": "system", "user": "human", "assistant": "gpt"}
            from_role = role_map.get(msg["role"], msg["role"])
            
            text_parts = []
            for content_item in msg["content"]:
                if content_item["type"] == "image":
                    images.append(content_item.get("image_path", ""))
                    text_parts.append("<image>")
                elif content_item["type"] == "text":
                    text_parts.append(content_item["value"])
                elif content_item["type"] == "reasoning":
                    text_parts.append(f"<reasoning>\n{content_item['value']}\n</reasoning>")
            
            conversations.append({
                "from": from_role,
                "value": "\n".join(text_parts)
            })
        
        formatted_data.append({
            "conversations": conversations,
            "images": images
        })
    
    return Dataset.from_list(formatted_data)


# 在 get_dataset 函数中添加分支
def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    **kwargs,
) -> Union["Dataset", "IterableDataset"]:
    
    # ... 原有代码 ...
    
    # 在加载数据集的地方添加
    for dataset_attr in get_dataset_list(data_args, training_args.do_train):
        if dataset_attr.load_from == "omnidocbench":  # 自定义标识
            dataset = load_omnidocbench_dataset(dataset_attr, data_args)
            all_datasets.append(dataset)
        else:
            # 原有的加载逻辑
            ...
```

### 步骤 2：配置 dataset_info.json

```json
{
  "omnidoc_dynamic": {
    "load_from": "omnidocbench",
    "dataset_name": "raw_omnidoc.jsonl",
    "formatting": "sharegpt"
  }
}
```

### 步骤 3：训练

```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen-VL-Chat \
  --dataset omnidoc_dynamic \
  --dataset_dir ./data \
  --output_dir ./output
```

---

## 方案 C：混合方式（最灵活）

### 创建独立的预处理脚本

```python
# preprocess_omnidoc.py
from omnibench import OmniDocConverter
import json

def preprocess_and_save(input_file, output_file):
    converter = OmniDocConverter()
    
    # 读取原始数据
    with open(input_file, 'r') as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    
    # 处理
    all_samples = []
    for page_data in raw_data:
        samples = converter.process_page(page_data)
        all_samples.extend(samples)
    
    # 转换为 ShareGPT 格式
    sharegpt_data = []
    for item in all_samples:
        conversations = []
        images = []
        
        for msg in item["messages"]:
            role_map = {"system": "system", "user": "human", "assistant": "gpt"}
            from_role = role_map[msg["role"]]
            
            text_parts = []
            for c in msg["content"]:
                if c["type"] == "image":
                    images.append(c.get("image_path", ""))
                    text_parts.append("<image>")
                elif c["type"] == "text":
                    text_parts.append(c["value"])
                elif c["type"] == "reasoning":
                    # 思维链用特殊标记包裹
                    text_parts.append(f"[REASONING]\n{c['value']}\n[/REASONING]")
            
            conversations.append({
                "from": from_role,
                "value": "\n".join(text_parts)
            })
        
        sharegpt_data.append({
            "conversations": conversations,
            "images": images
        })
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 处理完成：{len(sharegpt_data)} 条数据")

if __name__ == "__main__":
    preprocess_and_save(
        "data/omnidoc_raw.jsonl",
        "data/omnidoc_processed.json"
    )
```

然后正常使用 LLaMAFactory：

```bash
# 1. 预处理
python preprocess_omnidoc.py

# 2. 训练
llamafactory-cli train --dataset omnidoc_processed ...
```

---

## 思维链格式建议

### 选项 1：使用特殊标记

```
<reasoning>
Step 1: Locate the figure at [100, 300, 400, 600]
Step 2: Check parent_son relations
Step 3: Found caption at [100, 610, 400, 650]
</reasoning>

The caption is: "Figure 1: Comparison of ResNet and VGG accuracy."
```

### 选项 2：自然语言

```
Let me analyze this step by step:

First, I need to locate the figure at coordinates [100, 300, 400, 600].
Then, I'll check the layout relations for this element.
I found a 'parent_son' relation pointing to a caption block.
The caption text is: "Figure 1: Comparison of ResNet and VGG accuracy."
```

### 选项 3：结构化 JSON（训练时展开）

```json
{
  "reasoning_steps": [
    {"step": 1, "action": "locate", "target": "figure", "bbox": [100, 300, 400, 600]},
    {"step": 2, "action": "check_relation", "relation_type": "parent_son"},
    {"step": 3, "action": "extract_text", "result": "Figure 1: ..."}
  ],
  "final_answer": "The caption is: ..."
}
```

---

## 推荐工作流

1. **开发阶段**：使用方案 A（预处理），快速迭代
2. **优化阶段**：使用方案 C（独立脚本），便于调试
3. **生产阶段**：根据需求选择 A 或 B

---

## 常见问题

### Q1: 如何处理大规模数据集？
A: 使用流式处理 + IterableDataset

```python
from datasets import IterableDataset

def data_generator():
    with open("large_file.jsonl") as f:
        for line in f:
            page_data = json.loads(line)
            samples = converter.process_page(page_data)
            for sample in samples:
                yield format_sample(sample)

dataset = IterableDataset.from_generator(data_generator)
```

### Q2: 如何调试转换逻辑？
A: 先用小数据集测试

```python
# 只处理前 10 条
raw_data = raw_data[:10]
```

### Q3: 思维链会增加训练时间吗？
A: 会，因为序列更长。建议：
- 使用梯度累积
- 减小 batch size
- 使用 LoRA 等高效微调方法

---

## 下一步

1. 选择适合你的方案
2. 准备一个小数据集测试
3. 验证输出格式
4. 开始训练！

需要帮助？检查：
- `omnibench.py` - 你的转换逻辑
- `llamafactory_custom_loader.py` - 集成代码
- LLaMAFactory 文档：https://github.com/hiyouga/LLaMA-Factory
