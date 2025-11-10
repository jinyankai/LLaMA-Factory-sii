"""
OmniDocBench 增强版数据处理器
支持完整的 OmniDocBench 数据格式，包括：
- layout_dets: 布局元素（支持 text, latex, html, attribute, line_with_spans, merge_list）
- page_info: 页面信息（page_no, height, width, image_path, page_attribute）
- extra.relation: 关系标注（parent_son, truncated）

生成多种类型的思维链推理任务
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


# ==========================================
# 1. 增强的数据模型
# ==========================================

class LayoutElement:
    """增强的布局元素类，支持所有 OmniDocBench 字段"""
    
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        
        # 基础字段
        self.id = data.get("anno_id")
        self.category = data.get("category_type")
        self.poly = data.get("poly", [])
        self.ignore = data.get("ignore", False)
        # 确保 order 不为 None，用于排序
        self.order = data.get("order") if data.get("order") is not None else 0
        
        # 内容字段
        self.text = data.get("text", "")
        self.latex = data.get("latex", "")
        self.html = data.get("html", "")
        self.attribute = data.get("attribute", {})
        
        # 嵌套结构
        self.line_with_spans = data.get("line_with_spans", [])
        self.merge_list = data.get("merge_list", [])
        
        # 关系（后续填充）
        self.relations = []
    
    def get_bbox(self) -> List[float]:
        """获取边界框 [x1, y1, x2, y2]"""
        if len(self.poly) >= 8:
            x_coords = self.poly[0::2]
            y_coords = self.poly[1::2]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        return self.poly
    
    def get_center(self) -> tuple:
        """计算中心点"""
        bbox = self.get_bbox()
        if len(bbox) >= 4:
            return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return (0, 0)
    
    def get_bbox_str(self) -> str:
        """格式化的坐标字符串"""
        bbox = self.get_bbox()
        return f"[{', '.join(f'{int(x)}' for x in bbox)}]"
    
    def has_content(self) -> bool:
        """检查是否有实际内容"""
        return bool(self.text or self.latex or self.html)
    
    def get_full_content(self) -> str:
        """获取完整内容（优先级：text > latex > html）"""
        if self.text:
            return self.text
        if self.latex:
            return f"LaTeX: {self.latex}"
        if self.html:
            return f"HTML: {self.html}"
        return ""


class PageContext:
    """页面上下文，包含所有布局信息"""
    
    def __init__(self, page_data: Dict[str, Any]):
        self.page_info = page_data.get("page_info", {})
        self.page_no = self.page_info.get("page_no", 0)
        self.height = self.page_info.get("height", 0)
        self.width = self.page_info.get("width", 0)
        self.image_path = self.page_info.get("image_path", "")
        self.page_attribute = self.page_info.get("page_attribute", {})
        
        # 构建元素映射
        self.elements_map = {}
        for item in page_data.get("layout_dets", []):
            elem = LayoutElement(item)
            if not elem.ignore:  # 跳过标记为忽略的元素
                self.elements_map[elem.id] = elem
        
        # 解析关系
        self.relations = page_data.get("extra", {}).get("relation", [])
        self._build_relation_graph()
        
        # 按阅读顺序排序的元素列表
        self.ordered_elements = sorted(
            self.elements_map.values(),
            key=lambda e: e.order
        )
    
    def _build_relation_graph(self):
        """构建关系图"""
        for rel in self.relations:
            src_id = rel.get("source_anno_id")
            tgt_id = rel.get("target_anno_id")
            rel_type = rel.get("relation") or rel.get("relation_type")
            
            if src_id in self.elements_map and tgt_id in self.elements_map:
                src_elem = self.elements_map[src_id]
                tgt_elem = self.elements_map[tgt_id]
                
                src_elem.relations.append({
                    "target": tgt_elem,
                    "type": rel_type,
                    "role": "source"
                })
                tgt_elem.relations.append({
                    "target": src_elem,
                    "type": rel_type,
                    "role": "target"
                })
    
    def get_elements_by_category(self, category: str) -> List[LayoutElement]:
        """按类别获取元素"""
        return [e for e in self.elements_map.values() if e.category == category]
    
    def get_elements_by_order_range(self, start: int, end: int) -> List[LayoutElement]:
        """按阅读顺序范围获取元素"""
        return [e for e in self.ordered_elements if start <= e.order <= end]


# ==========================================
# 2. 任务生成器
# ==========================================

class BaseTask:
    """任务基类"""
    
    def __init__(self, context: PageContext):
        self.context = context
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        """生成任务，返回任务列表"""
        raise NotImplementedError


class CaptionRetrievalTask(BaseTask):
    """任务：图表标题检索"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        
        # 查找所有图表元素
        visual_categories = ["figure", "table", "chart", "equation"]
        
        for elem in self.context.elements_map.values():
            if elem.category in visual_categories:
                # 查找 parent_son 关系
                captions = [
                    r["target"] for r in elem.relations
                    if r["type"] == "parent_son" and r["role"] == "source"
                ]
                
                if captions:
                    caption = captions[0]
                    
                    question = f"What is the caption for the {elem.category} at position {elem.get_bbox_str()}?"
                    
                    reasoning_steps = [
                        f"1. Identify the {elem.category} element at {elem.get_bbox_str()}",
                        f"2. Check layout relations for element ID {elem.id}",
                        f"3. Found 'parent_son' relation to element ID {caption.id}",
                        f"4. Verify the target is a caption at {caption.get_bbox_str()}",
                        f"5. Extract the caption text"
                    ]
                    
                    tasks.append({
                        "type": "caption_retrieval",
                        "question": question,
                        "reasoning": "\n".join(reasoning_steps),
                        "answer": f"The caption is: \"{caption.get_full_content()}\""
                    })
        
        return tasks if tasks else None


class ReadingOrderTask(BaseTask):
    """任务：阅读顺序推理"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        
        # 只处理有足够元素的页面
        if len(self.context.ordered_elements) < 3:
            return None
        
        # 随机选择一个元素，询问其前后元素
        target_idx = random.randint(1, len(self.context.ordered_elements) - 2)
        target = self.context.ordered_elements[target_idx]
        prev_elem = self.context.ordered_elements[target_idx - 1]
        next_elem = self.context.ordered_elements[target_idx + 1]
        
        question = f"What comes after the {target.category} at {target.get_bbox_str()} in reading order?"
        
        reasoning_steps = [
            f"1. Locate the target {target.category} at {target.get_bbox_str()}",
            f"2. Check its reading order: {target.order}",
            f"3. Find the next element in sequence (order {next_elem.order})",
            f"4. Identify it as a {next_elem.category} at {next_elem.get_bbox_str()}",
            f"5. Extract its content"
        ]
        
        tasks.append({
            "type": "reading_order",
            "question": question,
            "reasoning": "\n".join(reasoning_steps),
            "answer": f"The next element is a {next_elem.category}: \"{next_elem.get_full_content()[:100]}...\""
        })
        
        return tasks


class TruncatedTextTask(BaseTask):
    """任务：跨页文本拼接"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        
        # 查找 truncated 关系
        for elem in self.context.elements_map.values():
            truncated_rels = [
                r for r in elem.relations
                if r["type"] == "truncated" and r["role"] == "source"
            ]
            
            if truncated_rels:
                target = truncated_rels[0]["target"]
                
                question = f"The text block at {elem.get_bbox_str()} appears to be truncated. Can you find its continuation?"
                
                reasoning_steps = [
                    f"1. Identify the truncated text block at {elem.get_bbox_str()}",
                    f"2. Check for 'truncated' relations from element ID {elem.id}",
                    f"3. Found continuation at element ID {target.id}",
                    f"4. Verify the continuation is at {target.get_bbox_str()}",
                    f"5. Concatenate the text segments"
                ]
                
                full_text = elem.get_full_content() + " " + target.get_full_content()
                
                tasks.append({
                    "type": "truncated_text",
                    "question": question,
                    "reasoning": "\n".join(reasoning_steps),
                    "answer": f"The complete text is: \"{full_text}\""
                })
        
        return tasks if tasks else None


class LayoutAnalysisTask(BaseTask):
    """任务：布局分析"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        
        # 统计页面布局
        category_counts = {}
        for elem in self.context.elements_map.values():
            category_counts[elem.category] = category_counts.get(elem.category, 0) + 1
        
        question = "What is the layout structure of this document page?"
        
        reasoning_steps = [
            "1. Scan all layout elements on the page",
            "2. Count elements by category type",
            "3. Analyze the document structure",
            "4. Summarize the layout composition"
        ]
        
        # 构建答案
        layout_summary = []
        for category, count in sorted(category_counts.items()):
            layout_summary.append(f"- {count} {category}(s)")
        
        answer = f"The page contains:\n" + "\n".join(layout_summary)
        
        tasks.append({
            "type": "layout_analysis",
            "question": question,
            "reasoning": "\n".join(reasoning_steps),
            "answer": answer
        })
        
        return tasks


class ContentExtractionTask(BaseTask):
    """任务：特定类型内容提取"""
    
    def generate(self) -> Optional[List[Dict[str, Any]]]:
        tasks = []
        
        # 提取所有标题
        titles = self.context.get_elements_by_category("title")
        if titles:
            title = titles[0]
            
            question = "What is the main title of this document page?"
            
            reasoning_steps = [
                "1. Search for 'title' category elements",
                f"2. Found title at {title.get_bbox_str()}",
                "3. Extract the title text",
                "4. Return the main title"
            ]
            
            tasks.append({
                "type": "content_extraction",
                "question": question,
                "reasoning": "\n".join(reasoning_steps),
                "answer": f"The title is: \"{title.get_full_content()}\""
            })
        
        # 提取公式
        equations = self.context.get_elements_by_category("equation")
        if equations and len(equations) > 0:
            eq = random.choice(equations)
            
            question = f"What is the mathematical equation at {eq.get_bbox_str()}?"
            
            reasoning_steps = [
                f"1. Locate the equation at {eq.get_bbox_str()}",
                "2. Check if LaTeX representation is available",
                "3. Extract the equation content",
                "4. Return the equation"
            ]
            
            tasks.append({
                "type": "content_extraction",
                "question": question,
                "reasoning": "\n".join(reasoning_steps),
                "answer": f"The equation is: {eq.latex or eq.text}"
            })
        
        return tasks if tasks else None


# ==========================================
# 3. 主转换器
# ==========================================

class OmniDocConverter:
    """OmniDocBench 数据转换器"""
    
    def __init__(self, task_types: Optional[List[str]] = None):
        """
        Args:
            task_types: 要生成的任务类型列表，None 表示全部
        """
        # 所有可用的任务生成器
        all_generators = {
            "caption_retrieval": CaptionRetrievalTask,
            "reading_order": ReadingOrderTask,
            "truncated_text": TruncatedTextTask,
            "layout_analysis": LayoutAnalysisTask,
            "content_extraction": ContentExtractionTask,
        }
        
        # 根据配置选择任务生成器
        if task_types is None:
            self.task_generators = list(all_generators.values())
        else:
            self.task_generators = [
                all_generators[t] for t in task_types if t in all_generators
            ]
    
    def process_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个页面，生成多个问答对"""
        context = PageContext(page_data)
        converted_samples = []
        
        for generator_cls in self.task_generators:
            generator = generator_cls(context)
            tasks = generator.generate()
            
            if tasks:
                for task in tasks:
                    sample = self._format_output(context, task)
                    converted_samples.append(sample)
        
        return converted_samples
    
    def _format_output(
        self,
        context: PageContext,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """格式化为最终输出格式"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "value": "You are a document analysis assistant. Analyze the layout and content step-by-step using the provided visual information."
                    }]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_path": context.image_path},
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
            ],
            "metadata": {
                "page_no": context.page_no,
                "task_type": task["type"]
            }
        }


# ==========================================
# 4. 命令行接口
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OmniDocBench 数据处理")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 文件")
    parser.add_argument("--tasks", nargs="+", help="任务类型列表")
    parser.add_argument("--max_pages", type=int, help="最大处理页数")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = OmniDocConverter(task_types=args.tasks)
    
    # 读取数据
    print(f"📖 读取数据: {args.input}")
    pages = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                pages.append(json.loads(line))
                if args.max_pages and len(pages) >= args.max_pages:
                    break
    
    print(f"✓ 读取了 {len(pages)} 页")
    
    # 处理
    print("🔄 处理中...")
    all_samples = []
    for page in pages:
        samples = converter.process_page(page)
        all_samples.extend(samples)
    
    print(f"✓ 生成了 {len(all_samples)} 个样本")
    
    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 保存到: {args.output}")
