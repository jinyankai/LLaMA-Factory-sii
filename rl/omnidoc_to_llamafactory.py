#!/usr/bin/env python3
"""
OmniDocBench 到 LLaMAFactory 的转换脚本

功能：
1. 读取 OmniDocBench 原始数据（JSONL 格式）
2. 应用思维链生成逻辑（基于 layout 标注）
3. 转换为 LLaMAFactory ShareGPT 格式
4. 生成对应的 dataset_info.json

使用方法：
    python omnidoc_to_llamafactory.py \
        --input data/omnidoc_raw.jsonl \
        --output data/omnidoc_processed.json \
        --dataset_name omnidoc_cot
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
try:
    from omnibench_enhanced import OmniDocConverter
except ImportError:
    from omnibench import OmniDocConverter


def convert_to_sharegpt(
    messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    将自定义消息格式转换为 LLaMAFactory ShareGPT 格式
    
    输入格式：
    {
        "messages": [
            {"role": "system", "content": [{"type": "text", "value": "..."}]},
            {"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]},
            {"role": "assistant", "content": [{"type": "reasoning", ...}, {"type": "text", ...}]}
        ]
    }
    
    输出格式：
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "<image>\n..."},
            {"from": "gpt", "value": "<reasoning>\n...\n</reasoning>\n..."}
        ],
        "images": ["path/to/image.jpg"]
    }
    """
    conversations = []
    images = []
    
    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "gpt"
    }
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        from_role = role_map.get(role, role)
        text_parts = []
        
        for content_item in content:
            content_type = content_item["type"]
            
            if content_type == "image":
                # 提取图像路径
                image_path = content_item.get("image_path", "")
                if image_path:
                    image_path = "images" + image_path
                    images.append(image_path)
                text_parts.append("<image>")
            
            elif content_type == "text":
                text_parts.append(content_item["value"])
            
            elif content_type == "reasoning":
                # 思维链用特殊标记包裹
                reasoning_text = content_item["value"]
                text_parts.append(f"<reasoning>\n{reasoning_text}\n</reasoning>")
        
        conversations.append({
            "from": from_role,
            "value": "\n".join(text_parts)
        })
    
    result = {"conversations": conversations}
    if images:
        result["images"] = images
    
    return result


def process_omnidoc_file(
    input_file: str,
    output_file: str,
    max_samples: int = None
) -> int:
    """
    处理 OmniDocBench 文件
    
    Args:
        input_file: 输入 JSONL 文件路径
        output_file: 输出 JSON 文件路径
        max_samples: 最大处理样本数（用于测试）
    
    Returns:
        处理的样本总数
    """
    print(f"📖 读取数据: {input_file}")
    
    # 读取原始数据
    raw_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # 检查文件格式
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON数组格式
            print("检测到JSON数组格式，正在解析...")
            data = json.load(f)
            raw_data = data[:max_samples] if max_samples else data
            if max_samples and len(data) > max_samples:
                print(f"⚠️  达到最大样本数限制: {max_samples}")
        else:
            # JSONL格式（每行一个JSON对象）
            print("检测到JSONL格式，正在逐行解析...")
            for idx, line in enumerate(f):
                if line.strip():
                    raw_data.append(json.loads(line))
                    if max_samples and len(raw_data) >= max_samples:
                        print(f"⚠️  达到最大样本数限制: {max_samples}")
                        break
    
    print(f"✓ 读取了 {len(raw_data)} 页数据")
    
    # 应用转换
    print("🔄 应用思维链转换...")
    converter = OmniDocConverter()
    all_samples = []
    
    for idx, page_data in enumerate(raw_data):
        try:
            samples = converter.process_page(page_data)
            all_samples.extend(samples)
            
            if (idx + 1) % 100 == 0:
                print(f"  处理进度: {idx + 1}/{len(raw_data)} 页")
        
        except Exception as e:
            print(f"⚠️  处理第 {idx} 页时出错: {e}")
            continue
    
    print(f"✓ 生成了 {len(all_samples)} 个问答对")
    
    # 转换为 ShareGPT 格式
    print("🔄 转换为 ShareGPT 格式...")
    sharegpt_data = []
    
    for sample in all_samples:
        try:
            sharegpt_sample = convert_to_sharegpt(sample["messages"])
            sharegpt_data.append(sharegpt_sample)
        except Exception as e:
            print(f"⚠️  转换样本时出错: {e}")
            continue
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 保存成功: {output_file}")
    print(f"   总计: {len(sharegpt_data)} 条训练数据")
    
    return len(sharegpt_data)


def create_dataset_info(
    dataset_name: str,
    file_name: str,
    output_dir: str
) -> None:
    """
    创建 dataset_info.json 文件
    
    Args:
        dataset_name: 数据集名称
        file_name: 数据文件名（相对路径）
        output_dir: 输出目录
    """
    dataset_info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }
    }
    
    info_path = Path(output_dir) / "dataset_info.json"
    
    # 如果文件已存在，合并
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            existing_info = json.load(f)
        existing_info.update(dataset_info)
        dataset_info = existing_info
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ dataset_info.json 已更新: {info_path}")


def preview_sample(output_file: str, num_samples: int = 2) -> None:
    """预览生成的样本"""
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"📋 数据预览（前 {num_samples} 条）")
    print(f"{'='*60}\n")
    
    for idx, sample in enumerate(data[:num_samples]):
        print(f"--- 样本 {idx + 1} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="将 OmniDocBench 数据转换为 LLaMAFactory 格式"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSON 文件路径"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="omnidoc_cot",
        help="数据集名称（用于 dataset_info.json）"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试）"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="预览生成的样本"
    )
    
    parser.add_argument(
        "--no_dataset_info",
        action="store_true",
        help="不生成 dataset_info.json"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("OmniDocBench → LLaMAFactory 转换工具")
    print("="*60)
    print()
    
    # 处理数据
    num_samples = process_omnidoc_file(
        args.input,
        args.output,
        args.max_samples
    )
    
    # 生成 dataset_info.json
    if not args.no_dataset_info:
        output_dir = Path(args.output).parent
        file_name = Path(args.output).name
        create_dataset_info(args.dataset_name, file_name, output_dir)
    
    # 预览
    if args.preview and num_samples > 0:
        preview_sample(args.output)
    
    print("\n" + "="*60)
    print("✅ 转换完成！")
    print("="*60)
    print(f"\n下一步：使用 LLaMAFactory 训练")
    print(f"  llamafactory-cli train \\")
    print(f"    --model_name_or_path Qwen/Qwen-VL-Chat \\")
    print(f"    --dataset {args.dataset_name} \\")
    print(f"    --dataset_dir {Path(args.output).parent} \\")
    print(f"    --output_dir ./output")
    print()


if __name__ == "__main__":
    main()
