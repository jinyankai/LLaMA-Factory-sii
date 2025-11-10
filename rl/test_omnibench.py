#!/usr/bin/env python3
"""
测试 OmniDocBench 处理流程

使用模拟数据测试完整的转换流程
"""

import json
from omnibench_enhanced import OmniDocConverter, PageContext


# 模拟真实的 OmniDocBench 数据
MOCK_DATA = {
    "layout_dets": [
        {
            "category_type": "title",
            "poly": [100.0, 50.0, 800.0, 50.0, 800.0, 100.0, 100.0, 100.0],
            "ignore": False,
            "order": 0,
            "anno_id": 0,
            "text": "Deep Learning for Document Understanding",
            "attribute": {}
        },
        {
            "category_type": "text_block",
            "poly": [100.0, 150.0, 800.0, 150.0, 800.0, 250.0, 100.0, 250.0],
            "ignore": False,
            "order": 1,
            "anno_id": 1,
            "text": "This paper presents a novel approach to document analysis using deep neural networks.",
            "attribute": {}
        },
        {
            "category_type": "figure",
            "poly": [100.0, 300.0, 500.0, 300.0, 500.0, 600.0, 100.0, 600.0],
            "ignore": False,
            "order": 2,
            "anno_id": 2,
            "text": "",
            "attribute": {}
        },
        {
            "category_type": "caption",
            "poly": [100.0, 610.0, 500.0, 610.0, 500.0, 650.0, 100.0, 650.0],
            "ignore": False,
            "order": 3,
            "anno_id": 3,
            "text": "Figure 1: Architecture of the proposed model showing the encoder-decoder structure.",
            "attribute": {}
        },
        {
            "category_type": "text_block",
            "poly": [100.0, 700.0, 800.0, 700.0, 800.0, 900.0, 100.0, 900.0],
            "ignore": False,
            "order": 4,
            "anno_id": 4,
            "text": "The model consists of three main components: a visual encoder, a text encoder, and",
            "attribute": {}
        },
        {
            "category_type": "text_block",
            "poly": [100.0, 950.0, 800.0, 950.0, 800.0, 1050.0, 100.0, 1050.0],
            "ignore": False,
            "order": 5,
            "anno_id": 5,
            "text": "a cross-modal fusion module that combines visual and textual features.",
            "attribute": {}
        },
        {
            "category_type": "equation",
            "poly": [200.0, 1100.0, 700.0, 1100.0, 700.0, 1150.0, 200.0, 1150.0],
            "ignore": False,
            "order": 6,
            "anno_id": 6,
            "text": "",
            "latex": "$L = \\alpha L_{cls} + \\beta L_{reg} + \\gamma L_{seg}$",
            "attribute": {}
        },
        {
            "category_type": "table",
            "poly": [100.0, 1200.0, 800.0, 1200.0, 800.0, 1500.0, 100.0, 1500.0],
            "ignore": False,
            "order": 7,
            "anno_id": 7,
            "text": "",
            "html": "<table><tr><th>Method</th><th>Accuracy</th></tr><tr><td>Ours</td><td>95.2%</td></tr></table>",
            "attribute": {}
        },
        {
            "category_type": "caption",
            "poly": [100.0, 1510.0, 800.0, 1510.0, 800.0, 1550.0, 100.0, 1550.0],
            "ignore": False,
            "order": 8,
            "anno_id": 8,
            "text": "Table 1: Comparison of different methods on the benchmark dataset.",
            "attribute": {}
        }
    ],
    "page_info": {
        "page_no": 1,
        "height": 1684,
        "width": 1200,
        "image_path": "data/sample_doc_page1.jpg",
        "page_attribute": {"document_type": "academic_paper"}
    },
    "extra": {
        "relation": [
            {
                "source_anno_id": 2,
                "target_anno_id": 3,
                "relation": "parent_son"
            },
            {
                "source_anno_id": 7,
                "target_anno_id": 8,
                "relation": "parent_son"
            },
            {
                "source_anno_id": 4,
                "target_anno_id": 5,
                "relation_type": "truncated"
            }
        ]
    }
}


def test_page_context():
    """测试 PageContext 解析"""
    print("="*60)
    print("测试 1: PageContext 解析")
    print("="*60)
    
    context = PageContext(MOCK_DATA)
    
    print(f"✓ 页面信息:")
    print(f"  - 页码: {context.page_no}")
    print(f"  - 尺寸: {context.width}x{context.height}")
    print(f"  - 图像: {context.image_path}")
    print(f"  - 元素数量: {len(context.elements_map)}")
    print(f"  - 关系数量: {len(context.relations)}")
    
    print(f"\n✓ 元素类型统计:")
    category_counts = {}
    for elem in context.elements_map.values():
        category_counts[elem.category] = category_counts.get(elem.category, 0) + 1
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")
    
    print(f"\n✓ 关系类型:")
    for rel in context.relations:
        src = context.elements_map[rel["source_anno_id"]]
        tgt = context.elements_map[rel["target_anno_id"]]
        rel_type = rel.get("relation") or rel.get("relation_type")
        print(f"  - {src.category}(ID:{src.id}) --[{rel_type}]--> {tgt.category}(ID:{tgt.id})")
    
    print()


def test_task_generation():
    """测试任务生成"""
    print("="*60)
    print("测试 2: 任务生成")
    print("="*60)
    
    converter = OmniDocConverter()
    samples = converter.process_page(MOCK_DATA)
    
    print(f"✓ 生成了 {len(samples)} 个任务\n")
    
    for idx, sample in enumerate(samples, 1):
        task_type = sample["metadata"]["task_type"]
        messages = sample["messages"]
        
        print(f"--- 任务 {idx}: {task_type} ---")
        
        # 提取问题
        user_msg = next(m for m in messages if m["role"] == "user")
        question = next(c["value"] for c in user_msg["content"] if c["type"] == "text")
        print(f"问题: {question}")
        
        # 提取推理
        assistant_msg = next(m for m in messages if m["role"] == "assistant")
        reasoning = next((c["value"] for c in assistant_msg["content"] if c["type"] == "reasoning"), None)
        if reasoning:
            print(f"推理:\n{reasoning}")
        
        # 提取答案
        answer = next((c["value"] for c in assistant_msg["content"] if c["type"] == "text"), None)
        if answer:
            print(f"答案: {answer}")
        
        print()


def test_sharegpt_conversion():
    """测试 ShareGPT 格式转换"""
    print("="*60)
    print("测试 3: ShareGPT 格式转换")
    print("="*60)
    
    from omnidoc_to_llamafactory import convert_to_sharegpt
    
    converter = OmniDocConverter()
    samples = converter.process_page(MOCK_DATA)
    
    if samples:
        sample = samples[0]
        sharegpt = convert_to_sharegpt(sample["messages"])
        
        print("✓ 转换后的格式:")
        print(json.dumps(sharegpt, indent=2, ensure_ascii=False))
    
    print()


def test_specific_tasks():
    """测试特定任务类型"""
    print("="*60)
    print("测试 4: 特定任务类型")
    print("="*60)
    
    # 只生成特定类型的任务
    task_types = ["caption_retrieval", "content_extraction"]
    converter = OmniDocConverter(task_types=task_types)
    samples = converter.process_page(MOCK_DATA)
    
    print(f"✓ 指定任务类型: {task_types}")
    print(f"✓ 生成了 {len(samples)} 个任务")
    
    for sample in samples:
        task_type = sample["metadata"]["task_type"]
        print(f"  - {task_type}")
    
    print()


def test_full_pipeline():
    """测试完整流程"""
    print("="*60)
    print("测试 5: 完整流程（保存文件）")
    print("="*60)
    
    # 创建测试数据文件
    test_input = "test_input.jsonl"
    test_output = "test_output.json"
    
    print(f"✓ 创建测试输入文件: {test_input}")
    with open(test_input, 'w', encoding='utf-8') as f:
        # 写入3页相同的数据
        for i in range(3):
            data = MOCK_DATA.copy()
            data["page_info"]["page_no"] = i + 1
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 运行转换
    print(f"✓ 运行转换...")
    from omnidoc_to_llamafactory import process_omnidoc_file
    
    num_samples = process_omnidoc_file(test_input, test_output, max_samples=3)
    
    print(f"✓ 生成了 {num_samples} 个训练样本")
    print(f"✓ 输出文件: {test_output}")
    
    # 读取并验证
    with open(test_output, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ 验证: 文件包含 {len(data)} 个样本")
    
    # 清理
    import os
    os.remove(test_input)
    print(f"✓ 清理测试文件")
    
    print()


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("OmniDocBench 处理流程测试")
    print("="*60 + "\n")
    
    try:
        test_page_context()
        test_task_generation()
        test_sharegpt_conversion()
        test_specific_tasks()
        test_full_pipeline()
        
        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n下一步:")
        print("1. 准备真实的 OmniDocBench 数据")
        print("2. 运行: python omnidoc_to_llamafactory.py --input your_data.jsonl --output output.json")
        print("3. 使用 LLaMAFactory 训练")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
