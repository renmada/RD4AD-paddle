# Linux端基础训练推理功能测试

Linux端基础训练推理功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称  | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|:-----:| :------: | :------: | :------: | :------: | :------------------: |
| RD4AD |   RD4AD    | 正常训练 |    -     |    -     |          -           |


- 推理相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的推理功能汇总如下，

| 算法名称 | 模型名称 | 模型类型 | device | batchsize | tensorrt | mkldnn | cpu多线程 |
| :------: | :------: | -------- | :----: |:---------:| :------: | :----: | :-------: |
|   RD4AD    |   RD4AD    | 正常模型 |  GPU   |    16     |    -     |   -    |     -     |


## 2. 测试流程

### 2.1 准备数据

少量数据在sample_data下

### 2.2 准备环境
- 框架：
  - paddlepaddle-gpu==2.3.1
- 使用如下命令安装依赖：

```bash
pip install -r requirements.txt
## 安装AutoLog（规范化日志输出工具）
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```

### 2.3 功能测试


测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_whole_infer
```

以`Linux GPU/CPU 基础训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/train_infer_python.txt  lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```
INFO 2022-07-24 15:30:39,874 autolog.py:264]  preprocess_time(ms): 87.1029, inference_time(ms): 2206.2488, postprocess_time(ms): 11.5511
[2022/07/24 15:30:39] root INFO:  preprocess_time(ms): 87.1029, inference_time(ms): 2206.2488, postprocess_time(ms): 11.5511
Outputs saved in ./test_tipc/output/RD4AD/norm_train_gpus_0_autocast_null
 Run successfully with command - python3 infer.py --data_dir sample_data --cls bottle --output_dir ./test_tipc/output/RD4AD/norm_train_gpus_0_autocast_null!  
```



## 3. 更多教程

本文档为功能测试用，更丰富的训练预测使用教程请参考：  

* [模型训练、预测、推理教程](../../README.md)  