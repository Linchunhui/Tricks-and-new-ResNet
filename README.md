# Tricks-and-new-ResNet
Bag of Tricks for Image Classification with Convolutional Neural Networks and a new ResNet.

A new [ResNet-D](https://github.com/Linchunhui/Tricks-and-new-ResNet/blob/master/ResNet-D.py) is here.  
And here is the [paper](https://arxiv.org/pdf/1812.01187.pdf).

# 论文笔记
我们在训练网络的时候经常会发现一模一样的网络总是达不到人家的效果，主要是因为他们还有一些`trick`。  
这篇文章主要就是列举了一些`tricks`以及对**ResNet**网络结构进行了一些调整，就将`ResNet-50`在`ImageNet`
的结果从**75.3%**提升到了**79.29**，甚至超过了`SE-ResNet-50`的**76.71**以及`DenseNet-201`的**79.29%**。
结果如下：
 <div align="center">![image](https://github.com/Linchunhui/Tricks-and-new-ResNet/blob/master/image/result.png)</div>

## 训练程序
