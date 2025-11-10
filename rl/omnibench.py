import json
import random
import os
from typing import List, Dict, Any, Optional

# ==========================================
# 1. 数据模型定义 (Data Models)
# ==========================================

class LayoutElement:
    """对原始layout_dets元素的封装，方便操作"""
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.id = data.get("anno_id")
        self.category = data.get("category_type")
        self.bbox = data.get("poly")
        self.text = data.get("text", "")
        self.relations = [] # 将会在后续步骤填充

    def get_center(self):
        """计算中心点，用于判断方位"""
        x_coords = self.bbox[0::2]
        y_coords = self.bbox[1::2]
        return sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords)

    def get_bbox_str(self):
        """返回格式化的坐标字符串，用于推理文本"""
        # 简化为 [x1, y1, x2, y2] 格式以便阅读，或者保留原始8点
        return f"{[int(x) for x in self.bbox]}"

class PageContext:
    """保存当前页面的所有上下文信息"""
    def __init__(self, page_data: Dict[str, Any]):
        self.page_info = page_data.get("page_info", {})
        self.elements_map = {item["anno_id"]: LayoutElement(item) for item in page_data.get("layout_dets", [])}
        self.relations = page_data.get("extra", {}).get("relation", [])
        self._build_relation_graph()

    def _build_relation_graph(self):
        """解析 relations，将其绑定到具体的 LayoutElement 对象上"""
        for rel in self.relations:
            src_id = rel["source_anno_id"]
            tgt_id = rel["target_anno_id"]
            rel_type = rel.get("relation") or rel.get("relation_type")
            
            if src_id in self.elements_map and tgt_id in self.elements_map:
                # 双向绑定，方便查询
                self.elements_map[src_id].relations.append({"target": self.elements_map[tgt_id], "type": rel_type, "role": "source"})
                self.elements_map[tgt_id].relations.append({"target": self.elements_map[src_id], "type": rel_type, "role": "target"})

# ==========================================
# 2. 推理任务工厂 (Task Factory)
# ==========================================
# 核心思想：不同的任务类型有不同的“提问模板”和“证据搜集逻辑”

class BaseTask:
    """所有推理任务的基类"""
    def __init__(self, context: PageContext):
        self.context = context

    def generate(self) -> Optional[Dict[str, Any]]:
        """返回一个包含 question, reasoning_steps, answer 的字典"""
        raise NotImplementedError

class CaptionRetrievalTask(BaseTask):
    """任务类型：给定图片/表格，询问其标题/图注"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        # 1. 遍历所有元素，寻找有 'parent_son' 关系的图/表
        for elem_id, elem in self.context.elements_map.items():
            if elem.category in ["figure", "table", "chart"]:
                # 寻找它的儿子（caption通常是son）
                captions = [r["target"] for r in elem.relations if r["type"] == "parent_son" and r["role"] == "source"]
                if captions:
                    caption_elem = captions[0]
                    
                    # 构建问题
                    question = f"Can you find the caption for the {elem.category} located at {elem.get_bbox_str()}?"
                    
                    # 构建结构化的推理步骤 (Evidence Chain)
                    reasoning_steps = [
                        f"I need to locate the caption for the {elem.category} at {elem.get_bbox_str()}.",
                        f"I will check the layout relations for this element (ID: {elem.id}).",
                        f"Found a 'parent_son' relation pointing to a text block (ID: {caption_elem.id}) located at {caption_elem.get_bbox_str()}.",
                        f"Verifying the category of the target block: it is '{caption_elem.category}'.",
                        f"Extracting the text from this block as the answer."
                    ]
                    
                    tasks.append({
                        "type": "caption_retrieval",
                        "question": question,
                        "reasoning": "\n".join(reasoning_steps), # 简单拼接，实际应用中可以用LLM润色
                        "answer": f"The caption is: \"{caption_elem.text}\""
                    })
        return tasks if tasks else None

class MainTopicTask(BaseTask):
    """任务类型：询问页面主旨（通常基于 Title）"""
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        # 寻找页面中最大的 Title 或者 order=0 的元素
        titles = [e for e in self.context.elements_map.values() if e.category == "title"]
        if titles:
            main_title = titles[0] # 简化处理，取第一个
            return [{
                "type": "main_topic",
                "question": "What is the main topic of this document page?",
                "reasoning": f"To find the main topic, I will look for the 'title' element.\nScanning the page layout, I found a prominent text block categorized as 'title' at {main_title.get_bbox_str()}.\nReading the content of this block to determine the topic.",
                "answer": f"According to the title, the main topic is '{main_title.text}'."
            }]
        return None

# ==========================================
# 3. 主转换管线 (Main Converter Pipeline)
# ==========================================

class OmniDocConverter:
    def __init__(self):
        # 注册所有可用的任务生成器
        self.task_generators = [
            CaptionRetrievalTask,
            MainTopicTask,
            # 在这里添加更多任务类型，如 ReadingOrderTask, TableValueTask 等
        ]

    def process_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        context = PageContext(page_data)
        converted_samples = []

        # 为当前页面生成所有可能的问答对
        for generator_cls in self.task_generators:
            generator = generator_cls(context)
            tasks = generator.generate()
            if tasks:
                for task in tasks:
                    # 格式化为目标结构
                    converted_samples.append(self._format_output(context, task))
        
        return converted_samples

    def _format_output(self, context: PageContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """转换为最终的 ReasonTool 格式"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text", 
                        "value": "You are a methodical document assistant. Analyze the layout and content step-by-step."
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        # 假设图片路径需要加上前缀
                        {"type": "image", "image_path": context.page_info.get("image_path", "")},
                        {"type": "text", "value": task["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "value": task["reasoning"]},
                        {"type": "text", "value": task["answer"]}
                    ]
                }
            ]
        }

# ==========================================
# 4. 运行示例 (Execution Demo)
# ==========================================

# 模拟一条 OmniDocBench 输入数据
mock_omnidoc_data = [
    {
        "layout_dets": [
            {
                "category_type": "title", "poly": [100, 100, 500, 100, 500, 150, 100, 150], "anno_id": 1,
                "text": "Analysis of Deep Learning Models"
            },
            {
                "category_type": "figure", "poly": [100, 300, 400, 300, 400, 600, 100, 600], "anno_id": 5,
                "text": "" # 图片本身无文本
            },
            {
                "category_type": "caption", "poly": [100, 610, 400, 610, 400, 650, 100, 650], "anno_id": 6,
                "text": "Figure 1: Comparison of ResNet and VGG accuracy."
            }
        ],
        "page_info": {"image_path": "train/images/doc_001.jpg", "height": 1000, "width": 800},
        "extra": {
            "relation": [
                {"source_anno_id": 5, "target_anno_id": 6, "relation": "parent_son"}
            ]
        }
    }
]

if __name__ == "__main__":
    print(">>> 开始转换流程...")
    converter = OmniDocConverter()
    all_converted_data = []
    
    for page_data in mock_omnidoc_data:
        # 处理每一页，可能会生成多个对话数据
        qa_pairs = converter.process_page(page_data)
        all_converted_data.extend(qa_pairs)

    # 打印结果预览
    print(f">>> 转换完成，共生成 {len(all_converted_data)} 条多模态推理数据。")
    for i, sample in enumerate(all_converted_data):
        print(f"\n--- Sample {i+1} ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False))