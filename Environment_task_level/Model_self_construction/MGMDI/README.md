# MMGAIN

这是一个 PyTorch框架下UCI数据集Letters的代码（代码仅以Letters为例），用于研究利用元学习做超参数自适应然后服务于填充模型。


## 运行

python main.py



主要超参数在 `config.py` 中修改，包括缺失率、缺失类型、训练轮数、学习率等。
project/
├── config          # 配置文件，存放模型的主要超参数 （这里的默认超参数设置可以跑但是应该效果不是最好的）
├── data_utils      # 对数据集进行读取划分和添加缺失等处理
├── evaluate        # 在测试集上的性能评估
├── losses          # 每个部分损失以及ARMSE评价指标计算
├── main            # 运行程序入口
├── missing_utils   # 给数据认为添加缺失具体实现
├── models          # 模型使用到的网络
├── train           # 模型训练文件
└── utils           # 工具方法设置随机种子等


两个代码都是运行main.py因该就可以，然后都是models和train是最主要介绍具体网络形式以及模型是如何展开训练的。