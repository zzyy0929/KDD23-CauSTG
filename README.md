# CauSTG 基于 Graph WaveNet for Deep Spatial-Temporal Graph Modeling 完成

## Train Commands
1. 基于划分数据训练k个模型
2. 对每个模型的参数做MinPooling，得到新的模型参数
3. 对新的模型参数做微调
```
sh train_env.sh
```


