# CauSTG 基于 Graph WaveNet for Deep Spatial-Temporal Graph Modeling 完成

## Train Commands
1. 基于划分数据训练k个模型
2. 对每个模型的参数做MinPooling，得到新的模型参数
3. 对新的模型参数做微调
```
sh train_env.sh
```
The implementation of "Maintaining the Status Qua: Capturing Invariant Relations for OOD Spatiotemporal Learning" accepted by SIGKDD conference 2023. This case is implemented on Metr-LA. 
Please unzip the file data.zip and __pycache__.zip.

