## 上海创智学院实训框架

### 训练方法

在examples/train_full/qwen3-full.yaml中修改配置（数据集/hyperparameters）

```bash
 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/qwen3-full.yaml 
```
开始训练SFT

### 强化学习框架
