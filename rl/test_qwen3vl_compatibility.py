#!/usr/bin/env python3
"""
Qwen3-VL 兼容性测试脚本

测试内容：
1. 模型加载
2. Processor 使用
3. 数据预处理
4. 前向传播
5. 布局感知损失计算

使用方法:
    python test_qwen3vl_compatibility.py --model_name Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from layout_aware_loss import LayoutAwareLoss, LayoutInfoExtractor


def test_model_loading(model_name: str):
    """测试 1: 模型加载"""
    print("\n" + "="*60)
    print("Test 1: Model Loading")
    print("="*60)
    
    try:
        # 加载模型
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载 processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Parameters: {model.num_parameters() / 1e9:.2f}B")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Dtype: {next(model.parameters()).dtype}")
        
        return model, processor
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        raise


def test_processor(processor):
    """测试 2: Processor 使用"""
    print("\n" + "="*60)
    print("Test 2: Processor Usage")
    print("="*60)
    
    try:
        # 测试消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the caption for this figure?"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The caption is: Example Figure"}
                ]
            }
        ]
        
        # 应用 chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        print(f"✓ Chat template applied successfully")
        print(f"  - Output length: {len(text)} chars")
        print(f"  - Preview: {text[:100]}...")
        
        # Tokenize
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        print(f"✓ Tokenization successful")
        print(f"  - Input IDs shape: {inputs['input_ids'].shape}")
        print(f"  - Attention mask shape: {inputs['attention_mask'].shape}")
        
        return inputs
        
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        raise


def test_data_preprocessing(processor):
    """测试 3: 数据预处理"""
    print("\n" + "="*60)
    print("Test 3: Data Preprocessing")
    print("="*60)
    
    try:
        # 模拟 ShareGPT 格式数据
        sample_data = {
            "conversations": [
                {
                    "from": "human",
                    "value": "What is the caption for this figure?"
                },
                {
                    "from": "gpt",
                    "value": "<reasoning>\n1. Locate figure with bbox [100, 200, 300, 400]\n2. Check parent_son relation\n3. Reading order: 1\n</reasoning>\n\nThe caption is: Example Figure"
                }
            ]
        }
        
        # 转换为 Qwen3-VL 格式
        messages = []
        for turn in sample_data["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            content = [{"type": "text", "text": turn["value"]}]
            messages.append({"role": role, "content": content})
        
        # 应用 template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 创建 labels
        inputs["labels"] = inputs["input_ids"].clone()
        
        print(f"✓ Data preprocessing successful")
        print(f"  - Input shape: {inputs['input_ids'].shape}")
        print(f"  - Labels shape: {inputs['labels'].shape}")
        
        return inputs
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        raise


def test_forward_pass(model, inputs):
    """测试 4: 前向传播"""
    print("\n" + "="*60)
    print("Test 4: Forward Pass")
    print("="*60)
    
    try:
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {outputs.loss.item():.4f}")
        print(f"  - Logits shape: {outputs.logits.shape}")
        
        return outputs
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_layout_loss(model, inputs, processor):
    """测试 5: 布局感知损失"""
    print("\n" + "="*60)
    print("Test 5: Layout-Aware Loss")
    print("="*60)
    
    try:
        # 创建损失函数
        loss_config = {
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.3,
            "delta": 0.2,
            "bbox_loss_type": "smooth_l1",
            "normalize_coords": True,
            "page_size": (1200, 1684),
        }
        
        layout_loss = LayoutAwareLoss(**loss_config)
        
        print(f"✓ Loss function created")
        print(f"  - Config: {loss_config}")
        
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 前向传播
        outputs = model(**inputs)
        
        # 解码文本（用于提取布局信息）
        decoded_texts = processor.batch_decode(
            inputs["input_ids"],
            skip_special_tokens=False
        )
        
        # 计算布局感知损失
        total_loss, loss_dict = layout_loss(
            outputs.logits,
            inputs["labels"],
            decoded_texts
        )
        
        print(f"✓ Layout-aware loss computed")
        print(f"  - Total loss: {total_loss.item():.4f}")
        print(f"  - Loss breakdown:")
        for key, value in loss_dict.items():
            if value is not None:
                print(f"    - {key}: {value.item():.4f}")
        
        # 测试布局信息提取
        print(f"\n✓ Testing layout extraction:")
        sample_text = decoded_texts[0]
        
        bboxes = LayoutInfoExtractor.extract_bboxes(sample_text)
        relations = LayoutInfoExtractor.extract_relations(sample_text)
        orders = LayoutInfoExtractor.extract_orders(sample_text)
        
        print(f"  - Bboxes found: {len(bboxes)}")
        print(f"  - Relations found: {len(relations)}")
        print(f"  - Orders found: {len(orders)}")
        
        if bboxes:
            print(f"  - Example bbox: {bboxes[0]}")
        if relations:
            print(f"  - Example relation: {relations[0]}")
        if orders:
            print(f"  - Example order: {orders[0]}")
        
        return total_loss, loss_dict
        
    except Exception as e:
        print(f"✗ Layout loss test failed: {e}")
        raise


def test_generation(model, processor):
    """测试 6: 生成测试"""
    print("\n" + "="*60)
    print("Test 6: Generation Test")
    print("="*60)
    
    try:
        # 准备输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is 2+2?"}
                ]
            }
        ]
        
        # 应用 template
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        print(f"✓ Generation successful")
        print(f"  - Input: What is 2+2?")
        print(f"  - Output: {output_text[0]}")
        
        return output_text
        
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen3-VL 兼容性测试")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL 模型名称"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="跳过生成测试（节省时间）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Qwen3-VL Compatibility Test Suite")
    print("="*60)
    print(f"Model: {args.model_name}")
    
    try:
        # 测试 1: 模型加载
        model, processor = test_model_loading(args.model_name)
        
        # 测试 2: Processor
        test_processor(processor)
        
        # 测试 3: 数据预处理
        inputs = test_data_preprocessing(processor)
        
        # 测试 4: 前向传播
        test_forward_pass(model, inputs)
        
        # 测试 5: 布局感知损失
        test_layout_loss(model, inputs, processor)
        
        # 测试 6: 生成（可选）
        if not args.skip_generation:
            test_generation(model, processor)
        
        # 总结
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nQwen3-VL is fully compatible with the layout-aware loss.")
        print("You can now proceed with training using:")
        print("  python train_qwen3vl_with_layout_loss.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ Test suite failed!")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. transformers >= 4.57.0 installed")
        print("  2. Model name is correct")
        print("  3. Sufficient GPU memory available")
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
