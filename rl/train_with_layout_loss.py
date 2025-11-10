#!/usr/bin/env python3
"""
使用布局感知损失训练 Qwen-VL 模型

这个脚本展示了如何：
1. 加载 OmniDocBench 处理后的数据
2. 使用自定义的布局感知损失函数
3. 训练多模态文档理解模型

使用方法：
    python train_with_layout_loss.py \
        --model_name Qwen/Qwen-VL-Chat \
        --data_file data/omnidoc_processed.json \
        --output_dir ./output \
        --num_epochs 3
"""

import os
import json
import argparse
from typing import Dict, List, Any
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, load_dataset

# 导入自定义损失函数
from layout_aware_loss import LayoutAwareLoss


# ==========================================
# 1. 数据处理
# ==========================================

def load_and_prepare_dataset(data_file: str, tokenizer, max_length: int = 2048):
    """加载并预处理数据集"""
    
    print(f"📖 加载数据: {data_file}")
    
    # 加载 JSON 数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 加载了 {len(data)} 条数据")
    
    # 转换为 HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # 预处理函数
    def preprocess_function(examples):
        """将 ShareGPT 格式转换为模型输入"""
        
        # 这里需要根据你的模型格式调整
        # 对于 Qwen-VL，通常需要特殊的格式化
        
        # 简化示例：拼接对话
        texts = []
        for conversations in examples["conversations"]:
            text = ""
            for turn in conversations:
                role = turn["from"]
                content = turn["value"]
                
                if role == "system":
                    text += f"System: {content}\n"
                elif role == "human":
                    text += f"Human: {content}\n"
                elif role == "gpt":
                    text += f"Assistant: {content}\n"
            
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建 labels（用于计算损失）
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    # 应用预处理
    print("🔄 预处理数据...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"✓ 预处理完成")
    
    return processed_dataset


# ==========================================
# 2. 自定义 Trainer
# ==========================================

class LayoutAwareTrainer(Trainer):
    """带有布局感知损失的 Trainer"""
    
    def __init__(self, *args, loss_config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化自定义损失函数
        self.layout_loss_fn = LayoutAwareLoss(**(loss_config or {}))
        
        # 控制是否解码文本（训练时关闭以提速）
        self.decode_for_layout_loss = False
        
        print("✓ 使用布局感知损失函数")
        print(f"  - 语言建模权重: {self.layout_loss_fn.alpha}")
        print(f"  - 边界框权重: {self.layout_loss_fn.beta}")
        print(f"  - 关系权重: {self.layout_loss_fn.gamma}")
        print(f"  - 顺序权重: {self.layout_loss_fn.delta}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算"""
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # 解码文本（可选，用于布局损失）
        pred_texts = None
        target_texts = None
        
        if self.decode_for_layout_loss:
            # 获取预测
            pred_ids = torch.argmax(logits, dim=-1)
            
            # 解码
            pred_texts = self.tokenizer.batch_decode(
                pred_ids,
                skip_special_tokens=True
            )
            
            # 解码目标（忽略 -100）
            target_ids = labels.clone()
            target_ids[target_ids == -100] = self.tokenizer.pad_token_id
            target_texts = self.tokenizer.batch_decode(
                target_ids,
                skip_special_tokens=True
            )
        
        # 计算损失
        loss_dict = self.layout_loss_fn(
            logits=logits,
            labels=labels,
            pred_texts=pred_texts,
            target_texts=target_texts
        )
        
        # 记录各项损失
        self.log({
            "lm_loss": loss_dict["lm_loss"].item(),
            "bbox_loss": loss_dict["bbox_loss"].item(),
            "relation_loss": loss_dict["relation_loss"].item(),
            "order_loss": loss_dict["order_loss"].item(),
        })
        
        loss = loss_dict["loss"]
        
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 3. 主训练函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="使用布局感知损失训练模型")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-VL-Chat",
                        help="模型名称或路径")
    parser.add_argument("--data_file", type=str, required=True,
                        help="训练数据文件（JSON 格式）")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    
    # 损失函数参数
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="语言建模损失权重")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="边界框损失权重")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="关系损失权重")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="顺序损失权重")
    parser.add_argument("--bbox_loss_type", type=str, default="smooth_l1",
                        choices=["smooth_l1", "iou"],
                        help="边界框损失类型")
    
    # 其他参数
    parser.add_argument("--fp16", action="store_true",
                        help="使用混合精度训练")
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    print("="*60)
    print("布局感知损失训练脚本")
    print("="*60)
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_file}")
    print(f"输出: {args.output_dir}")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 1. 加载模型和分词器
    print("\n📦 加载模型和分词器...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✓ 模型加载完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 2. 加载数据
    dataset = load_and_prepare_dataset(
        args.data_file,
        tokenizer,
        max_length=args.max_length
    )
    
    # 划分训练集和验证集
    if args.eval_split > 0:
        split_dataset = dataset.train_test_split(
            test_size=args.eval_split,
            seed=args.seed
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"✓ 数据划分: 训练 {len(train_dataset)} / 验证 {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"✓ 训练数据: {len(train_dataset)}")
    
    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        report_to="tensorboard",
        seed=args.seed,
    )
    
    # 4. 损失函数配置
    loss_config = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "bbox_loss_type": args.bbox_loss_type,
        "normalize_coords": True,
        "page_size": (1200, 1684),  # 根据你的数据调整
    }
    
    # 5. 创建 Trainer
    print("\n🚀 初始化 Trainer...")
    
    trainer = LayoutAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_config=loss_config,
    )
    
    # 6. 训练
    print("\n🎯 开始训练...")
    print(f"  总步数: {len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs}")
    print(f"  有效批次大小: {args.batch_size * args.gradient_accumulation_steps}")
    print()
    
    train_result = trainer.train()
    
    # 7. 保存模型
    print("\n💾 保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print(f"模型保存在: {args.output_dir}")
    print(f"日志保存在: {args.output_dir}/logs")
    print("\n查看训练日志:")
    print(f"  tensorboard --logdir {args.output_dir}/logs")
    print()


if __name__ == "__main__":
    main()
