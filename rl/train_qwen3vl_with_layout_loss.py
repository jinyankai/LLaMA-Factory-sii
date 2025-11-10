#!/usr/bin/env python3
"""
Qwen3-VL 布局感知损失训练脚本

支持 Qwen3-VL 系列模型的完整训练流程，包括：
- 自动布局信息提取
- 多任务损失函数（LM + BBox + Relation + Order）
- Flash Attention 2 加速
- Gradient Checkpointing 节省显存

使用方法:
    python train_qwen3vl_with_layout_loss.py \\
        --model_name Qwen/Qwen3-VL-8B-Instruct \\
        --data_file data/omnidoc_processed.json \\
        --output_dir ./output \\
        --num_epochs 3 \\
        --batch_size 2
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)

# 导入布局感知损失
from layout_aware_loss import LayoutAwareLoss, LayoutAwareTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL 布局感知训练")
    
    # 模型参数
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL 模型名称或路径"
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=True,
        help="使用 Flash Attention 2（推荐）"
    )
    parser.add_argument(
        "--use_fp8",
        action="store_true",
        help="使用 FP8 量化版本"
    )
    
    # 数据参数
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="训练数据文件（JSON 格式）"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="验证数据文件（可选）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大序列长度"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_qwen3vl",
        help="输出目录"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="每设备批次大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="学习率"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="预热比例"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="使用梯度检查点节省显存"
    )
    
    # 损失函数参数
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="语言模型损失权重"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="边界框损失权重"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="关系损失权重"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="顺序损失权重"
    )
    parser.add_argument(
        "--bbox_loss_type",
        type=str,
        default="smooth_l1",
        choices=["smooth_l1", "iou"],
        help="边界框损失类型"
    )
    parser.add_argument(
        "--normalize_coords",
        action="store_true",
        default=True,
        help="归一化坐标"
    )
    parser.add_argument(
        "--page_width",
        type=int,
        default=1200,
        help="页面宽度（用于归一化）"
    )
    parser.add_argument(
        "--page_height",
        type=int,
        default=1684,
        help="页面高度（用于归一化）"
    )
    
    # 其他参数
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="日志记录步数"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="保存检查点步数"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="评估步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def load_model_and_processor(args):
    """加载 Qwen3-VL 模型和处理器"""
    
    print(f"Loading model: {args.model_name}")
    
    # 确定模型名称
    model_name = args.model_name
    if args.use_fp8 and "FP8" not in model_name:
        model_name = model_name.replace("-Instruct", "-Instruct-FP8")
        print(f"Using FP8 version: {model_name}")
    
    # 加载模型
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Flash Attention 2
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Gradient Checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # 加载 Processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # 配置图像处理器（控制 token 数量）
    processor.image_processor.size = {
        "longest_edge": 1280 * 32 * 32,  # 最大 1280 tokens
        "shortest_edge": 256 * 32 * 32,  # 最小 256 tokens
    }
    
    print(f"Model loaded: {model.num_parameters() / 1e9:.2f}B parameters")
    
    return model, processor


def preprocess_function(examples, processor, max_length=2048):
    """
    预处理函数：将 ShareGPT 格式转换为 Qwen3-VL 格式
    
    输入格式:
    {
        "conversations": [
            {"from": "human", "value": "<image>\\nWhat is the caption?"},
            {"from": "gpt", "value": "The caption is: ..."}
        ],
        "images": ["path/to/image.jpg"]
    }
    """
    
    texts = []
    
    for conversations, images in zip(
        examples["conversations"],
        examples.get("images", [[]] * len(examples["conversations"]))
    ):
        # 构建 Qwen3-VL 消息格式
        messages = []
        
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = []
            
            # 处理图像（仅在用户消息中）
            if images and turn["from"] == "human":
                for img_path in images:
                    content.append({
                        "type": "image",
                        "image": img_path
                    })
            
            # 处理文本（移除 <image> 标记）
            text = turn["value"].replace("<image>", "").strip()
            if text:
                content.append({
                    "type": "text",
                    "text": text
                })
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # 应用 chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize
    model_inputs = processor(
        text=texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    
    # 创建 labels（复制 input_ids）
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # 将 padding token 的 label 设为 -100（忽略损失）
    padding_mask = model_inputs["attention_mask"] == 0
    model_inputs["labels"][padding_mask] = -100
    
    return model_inputs


def load_and_preprocess_data(args, processor):
    """加载并预处理数据"""
    
    print(f"Loading data from: {args.data_file}")
    
    # 加载数据
    data_files = {"train": args.data_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    
    dataset = load_dataset("json", data_files=data_files)
    
    print(f"Train samples: {len(dataset['train'])}")
    if "validation" in dataset:
        print(f"Validation samples: {len(dataset['validation'])}")
    
    # 预处理
    def preprocess_wrapper(examples):
        return preprocess_function(examples, processor, args.max_length)
    
    processed_dataset = dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing data"
    )
    
    return processed_dataset


def main():
    """主函数"""
    
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 1. 加载模型和处理器
    model, processor = load_model_and_processor(args)
    
    # 2. 加载和预处理数据
    processed_dataset = load_and_preprocess_data(args, processor)
    
    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,  # Qwen3-VL 推荐使用 bf16
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if args.val_file else "no",
        eval_steps=args.eval_steps if args.val_file else None,
        load_best_model_at_end=True if args.val_file else False,
        metric_for_best_model="loss" if args.val_file else None,
        greater_is_better=False,
        report_to="tensorboard",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 4. 损失函数配置
    loss_config = {
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "bbox_loss_type": args.bbox_loss_type,
        "normalize_coords": args.normalize_coords,
        "page_size": (args.page_width, args.page_height),
    }
    
    print("\n" + "="*50)
    print("Loss Configuration:")
    for key, value in loss_config.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # 5. 创建 Trainer（带布局感知损失）
    trainer = LayoutAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("validation"),
        tokenizer=processor.tokenizer,
        loss_config=loss_config,
    )
    
    # 6. 训练
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    train_result = trainer.train()
    
    # 7. 保存模型
    print("\n" + "="*50)
    print("Saving model...")
    print("="*50 + "\n")
    
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {os.path.join(args.output_dir, 'final_model')}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
