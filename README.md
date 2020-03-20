# imgCls

把bmg图片文件分别放入A、B、C分类文件夹中

执行processing.py 生成训练集和测试集的.npy文件

执行train.py，进行训练和预测，得到预测精确度

可以做的改进：

1保存模型，分离训练和测试为两个脚本

2在processing_data中，用opencv对图片进行旋转，增加数据量

3在processing_data中，目前是把输入的图片压成了32*32的size，这样损失了很多细节，尝试增加这个size，但也要考虑硬件是否能够增加，可以逐步增加。

4增加size的话，train中的模型需要修改，使输入输出能够对应

5增加层数
