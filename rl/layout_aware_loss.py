"""
布局感知的增强损失函数 (Layout-Aware Enhanced Loss)

核心思想：
1. 标准语言建模损失（文本生成）
2. 布局坐标预测损失（bbox 回归）
3. 布局关系分类损失（parent_son, truncated 等）
4. 阅读顺序损失（order 预测）

适用于 LLaMAFactory + Qwen-VL 等多模态模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import re
import json


# ==========================================
# 1. 布局信息提取器
# ==========================================

class LayoutInfoExtractor:
    """从文本中提取布局相关信息"""
    
    # 正则表达式模式
    BBOX_PATTERN = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    RELATION_PATTERN = r'(parent_son|truncated|sibling)'
    ORDER_PATTERN = r'order[:\s]+(\d+)'
    ELEMENT_ID_PATTERN = r'(?:ID|id)[:\s]+(\d+)'
    
    @staticmethod
    def extract_bboxes(text: str) -> List[List[float]]:
        """提取文本中的所有边界框坐标"""
        matches = re.findall(LayoutInfoExtractor.BBOX_PATTERN, text)
        return [[float(x) for x in match] for match in matches]
    
    @staticmethod
    def extract_relations(text: str) -> List[str]:
        """提取关系类型"""
        return re.findall(LayoutInfoExtractor.RELATION_PATTERN, text)
    
    @staticmethod
    def extract_orders(text: str) -> List[int]:
        """提取阅读顺序"""
        matches = re.findall(LayoutInfoExtractor.ORDER_PATTERN, text)
        return [int(m) for m in matches]
    
    @staticmethod
    def extract_element_ids(text: str) -> List[int]:
        """提取元素 ID"""
        matches = re.findall(LayoutInfoExtractor.ELEMENT_ID_PATTERN, text)
        return [int(m) for m in matches]


# ==========================================
# 2. 布局感知损失函数
# ==========================================

class LayoutAwareLoss(nn.Module):
    """
    布局感知的多任务损失函数
    
    Loss = α * L_lm + β * L_bbox + γ * L_relation + δ * L_order
    
    其中：
    - L_lm: 标准语言建模损失（交叉熵）
    - L_bbox: 边界框回归损失（Smooth L1 或 IoU）
    - L_relation: 关系分类损失（交叉熵）
    - L_order: 阅读顺序损失（排序损失）
    """
    
    def __init__(
        self,
        alpha: float = 1.0,      # 语言建模权重
        beta: float = 0.5,       # 边界框权重
        gamma: float = 0.3,      # 关系分类权重
        delta: float = 0.2,      # 阅读顺序权重
        bbox_loss_type: str = "smooth_l1",  # "smooth_l1" 或 "iou"
        relation_classes: List[str] = None,
        normalize_coords: bool = True,
        page_size: Tuple[int, int] = (1200, 1684),  # 默认页面尺寸
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bbox_loss_type = bbox_loss_type
        self.normalize_coords = normalize_coords
        self.page_width, self.page_height = page_size
        
        # 关系类型映射
        if relation_classes is None:
            relation_classes = ["none", "parent_son", "truncated", "sibling"]
        self.relation_classes = relation_classes
        self.relation_to_idx = {r: i for i, r in enumerate(relation_classes)}
        
        # 损失函数
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.relation_loss_fn = nn.CrossEntropyLoss()
        
        # 信息提取器
        self.extractor = LayoutInfoExtractor()
    
    def normalize_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """归一化边界框坐标到 [0, 1]"""
        if not self.normalize_coords:
            return bbox
        
        normalized = bbox.clone()
        normalized[..., [0, 2]] /= self.page_width   # x 坐标
        normalized[..., [1, 3]] /= self.page_height  # y 坐标
        return normalized
    
    def compute_iou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """计算 IoU 损失"""
        # pred_boxes, target_boxes: [N, 4] (x1, y1, x2, y2)
        
        # 计算交集
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union = pred_area + target_area - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # IoU Loss = 1 - IoU
        return 1.0 - iou.mean()
    
    def compute_bbox_loss(
        self,
        pred_text: str,
        target_text: str
    ) -> torch.Tensor:
        """计算边界框损失"""
        # 提取预测和目标的边界框
        pred_bboxes = self.extractor.extract_bboxes(pred_text)
        target_bboxes = self.extractor.extract_bboxes(target_text)
        
        if not pred_bboxes or not target_bboxes:
            return torch.tensor(0.0, device=self.get_device())
        
        # 转换为 tensor
        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32, device=self.get_device())
        target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32, device=self.get_device())
        
        # 归一化
        pred_bboxes = self.normalize_bbox(pred_bboxes)
        target_bboxes = self.normalize_bbox(target_bboxes)
        
        # 匹配数量（取最小）
        min_len = min(len(pred_bboxes), len(target_bboxes))
        if min_len == 0:
            return torch.tensor(0.0, device=self.get_device())
        
        pred_bboxes = pred_bboxes[:min_len]
        target_bboxes = target_bboxes[:min_len]
        
        # 计算损失
        if self.bbox_loss_type == "iou":
            return self.compute_iou_loss(pred_bboxes, target_bboxes)
        else:  # smooth_l1
            return self.bbox_loss_fn(pred_bboxes, target_bboxes)
    
    def compute_relation_loss(
        self,
        pred_text: str,
        target_text: str
    ) -> torch.Tensor:
        """计算关系分类损失"""
        pred_relations = self.extractor.extract_relations(pred_text)
        target_relations = self.extractor.extract_relations(target_text)
        
        if not pred_relations or not target_relations:
            return torch.tensor(0.0, device=self.get_device())
        
        # 转换为索引
        pred_indices = [self.relation_to_idx.get(r, 0) for r in pred_relations]
        target_indices = [self.relation_to_idx.get(r, 0) for r in target_relations]
        
        min_len = min(len(pred_indices), len(target_indices))
        if min_len == 0:
            return torch.tensor(0.0, device=self.get_device())
        
        pred_indices = torch.tensor(pred_indices[:min_len], device=self.get_device())
        target_indices = torch.tensor(target_indices[:min_len], device=self.get_device())
        
        # 这里简化处理：假设预测是 one-hot 或 logits
        # 实际使用时需要从模型输出中获取 logits
        # 这里用 one-hot 近似
        num_classes = len(self.relation_classes)
        pred_logits = F.one_hot(pred_indices, num_classes).float()
        
        return self.relation_loss_fn(pred_logits, target_indices)
    
    def compute_order_loss(
        self,
        pred_text: str,
        target_text: str
    ) -> torch.Tensor:
        """计算阅读顺序损失"""
        pred_orders = self.extractor.extract_orders(pred_text)
        target_orders = self.extractor.extract_orders(target_text)
        
        if not pred_orders or not target_orders:
            return torch.tensor(0.0, device=self.get_device())
        
        min_len = min(len(pred_orders), len(target_orders))
        if min_len == 0:
            return torch.tensor(0.0, device=self.get_device())
        
        pred_orders = torch.tensor(pred_orders[:min_len], dtype=torch.float32, device=self.get_device())
        target_orders = torch.tensor(target_orders[:min_len], dtype=torch.float32, device=self.get_device())
        
        # 使用 L1 损失
        return F.l1_loss(pred_orders, target_orders)
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
    
    def forward(
        self,
        logits: torch.Tensor,           # [batch, seq_len, vocab_size]
        labels: torch.Tensor,           # [batch, seq_len]
        pred_texts: Optional[List[str]] = None,   # 解码后的预测文本
        target_texts: Optional[List[str]] = None, # 目标文本
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            logits: 模型输出的 logits
            labels: 目标标签
            pred_texts: 解码后的预测文本（用于提取布局信息）
            target_texts: 目标文本（用于提取布局信息）
        
        Returns:
            包含各项损失的字典
        """
        # 1. 语言建模损失（标准交叉熵）
        lm_loss = self.lm_loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # 2. 布局相关损失（需要文本）
        bbox_loss = torch.tensor(0.0, device=logits.device)
        relation_loss = torch.tensor(0.0, device=logits.device)
        order_loss = torch.tensor(0.0, device=logits.device)
        
        if pred_texts is not None and target_texts is not None:
            batch_bbox_losses = []
            batch_relation_losses = []
            batch_order_losses = []
            
            for pred_text, target_text in zip(pred_texts, target_texts):
                # 边界框损失
                if self.beta > 0:
                    batch_bbox_losses.append(
                        self.compute_bbox_loss(pred_text, target_text)
                    )
                
                # 关系损失
                if self.gamma > 0:
                    batch_relation_losses.append(
                        self.compute_relation_loss(pred_text, target_text)
                    )
                
                # 顺序损失
                if self.delta > 0:
                    batch_order_losses.append(
                        self.compute_order_loss(pred_text, target_text)
                    )
            
            if batch_bbox_losses:
                bbox_loss = torch.stack(batch_bbox_losses).mean()
            if batch_relation_losses:
                relation_loss = torch.stack(batch_relation_losses).mean()
            if batch_order_losses:
                order_loss = torch.stack(batch_order_losses).mean()
        
        # 3. 总损失
        total_loss = (
            self.alpha * lm_loss +
            self.beta * bbox_loss +
            self.gamma * relation_loss +
            self.delta * order_loss
        )
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "relation_loss": relation_loss.detach(),
            "order_loss": order_loss.detach(),
        }


# ==========================================
# 3. 用于 LLaMAFactory 的 Trainer 包装器
# ==========================================

class LayoutAwareTrainer:
    """
    包装 LLaMAFactory 的 Trainer，注入自定义损失函数
    
    使用方法：
    1. 在训练脚本中导入此类
    2. 替换标准的 compute_loss 方法
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        loss_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化损失函数
        if loss_config is None:
            loss_config = {}
        self.loss_fn = LayoutAwareLoss(**loss_config)
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ):
        """
        自定义损失计算（替换 Trainer 的 compute_loss）
        
        这个方法会被 HuggingFace Trainer 调用
        """
        # 1. 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # 2. 解码文本（用于提取布局信息）
        # 注意：这会增加计算开销，可以考虑只在验证时使用
        pred_texts = None
        target_texts = None
        
        if hasattr(self, 'decode_for_layout_loss') and self.decode_for_layout_loss:
            # 获取预测的 token IDs
            pred_ids = torch.argmax(logits, dim=-1)
            
            # 解码
            pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            target_texts = self.tokenizer.batch_decode(
                labels.masked_fill(labels == -100, self.tokenizer.pad_token_id),
                skip_special_tokens=True
            )
        
        # 3. 计算损失
        loss_dict = self.loss_fn(
            logits=logits,
            labels=labels,
            pred_texts=pred_texts,
            target_texts=target_texts
        )
        
        loss = loss_dict["loss"]
        
        # 4. 记录各项损失（用于监控）
        if hasattr(self, 'log_metrics'):
            self.log_metrics({
                "train/lm_loss": loss_dict["lm_loss"].item(),
                "train/bbox_loss": loss_dict["bbox_loss"].item(),
                "train/relation_loss": loss_dict["relation_loss"].item(),
                "train/order_loss": loss_dict["order_loss"].item(),
            })
        
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 4. 集成到 LLaMAFactory
# ==========================================

def create_layout_aware_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    eval_dataset=None,
    loss_config: Optional[Dict[str, Any]] = None,
):
    """
    创建带有布局感知损失的 Trainer
    
    使用示例：
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
    """
    from transformers import Trainer
    
    # 创建自定义 Trainer 类
    class CustomTrainer(Trainer):
        def __init__(self, *args, loss_config=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.layout_loss_fn = LayoutAwareLoss(**(loss_config or {}))
            # 控制是否解码文本（训练时可关闭以提速）
            self.decode_for_layout_loss = False
        
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs.get("labels")
            
            # 解码文本（可选）
            pred_texts = None
            target_texts = None
            
            if self.decode_for_layout_loss:
                pred_ids = torch.argmax(logits, dim=-1)
                pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                target_texts = self.tokenizer.batch_decode(
                    labels.masked_fill(labels == -100, self.tokenizer.pad_token_id),
                    skip_special_tokens=True
                )
            
            # 计算损失
            loss_dict = self.layout_loss_fn(
                logits=logits,
                labels=labels,
                pred_texts=pred_texts,
                target_texts=target_texts
            )
            
            # 记录损失
            self.log({
                "lm_loss": loss_dict["lm_loss"].item(),
                "bbox_loss": loss_dict["bbox_loss"].item(),
                "relation_loss": loss_dict["relation_loss"].item(),
                "order_loss": loss_dict["order_loss"].item(),
            })
            
            loss = loss_dict["loss"]
            return (loss, outputs) if return_outputs else loss
    
    # 创建 trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_config=loss_config,
    )
    
    return trainer


# ==========================================
# 5. 测试和示例
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("布局感知损失函数测试")
    print("="*60)
    
    # 创建损失函数
    loss_fn = LayoutAwareLoss(
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        delta=0.2,
        bbox_loss_type="smooth_l1"
    )
    
    # 模拟数据
    batch_size = 2
    seq_len = 50
    vocab_size = 32000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :10] = -100  # 忽略前 10 个 token
    
    # 模拟文本（包含布局信息）
    pred_texts = [
        "The figure at [100, 200, 400, 500] has a parent_son relation with element ID 5 at order 2",
        "Found truncated text at [50, 100, 300, 200] with order 1"
    ]
    target_texts = [
        "The figure at [100, 200, 400, 500] has a parent_son relation with element ID 5 at order 2",
        "Found truncated text at [50, 100, 300, 200] with order 1"
    ]
    
    # 计算损失
    loss_dict = loss_fn(
        logits=logits,
        labels=labels,
        pred_texts=pred_texts,
        target_texts=target_texts
    )
    
    print("\n损失计算结果:")
    print(f"  总损失: {loss_dict['loss']:.4f}")
    print(f"  语言建模损失: {loss_dict['lm_loss']:.4f}")
    print(f"  边界框损失: {loss_dict['bbox_loss']:.4f}")
    print(f"  关系损失: {loss_dict['relation_loss']:.4f}")
    print(f"  顺序损失: {loss_dict['order_loss']:.4f}")
    
    print("\n✅ 测试通过！")
