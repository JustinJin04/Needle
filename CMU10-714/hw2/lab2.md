# lab2 记录

## 整体设计


最底层：基本算子(ops)，Tensor类

* 提供抽象：

  * 对tensor调用ops自动构建实现计算图，如：

    ```python
    loss = f(input_tensor) # f为用一些列ops复合而成的计算关系
    loss.backward()	# 自动计算出loss相对f中每个ops输出的梯度（存放在ops结构体内部）
    ```

  * 对于不参与loss计算的tensor可以通过detach()从计算图中分离。当前epoch结束以后前一轮的activation memory可以正常释放(否则可能某些计算图之外的tensor一直存在于内存中，其input连着activation导致该部分activation无法被释放)

* 实现细节：

  * 每个tensor要么是leaf node，要么是通过input tensor和ops计算出来的
  * 在前向传播的时候动态构建计算图（tensor作为结点，ops接受tensor作为参数产生新的tensor）
  * 反向传播时，根据计算图的反拓扑排序（**数学证明？？？？**）的顺序求出loss相对每个tensor的梯度



上层：网络模型(model)，优化器optimizer

* 提供抽象：

  ```python
  model = Resnet(*parameters)
  opt = optimizer(model.parameters(),*parameters)	# self.parameters()为了区分模型内部中还有一些不参与计算图的常量
  pred = model(input_tensor)
  loss = loss_fn(pred,label)
  loss.backward()
  opt.step()
  ```

* 实现细节：

  * model: 由模型参数(model.parameters())和一系列ops组成
    * 注：还有可能有类似norm层内存储的一些不在计算图中的常量，注意不要直接引用计算图中的activation计算
  * optimizer: 接管model.parameters()，对其中的每个tensor实施更新梯度



显存分析

* parameters：在创建模型的时候分配，如Linear层的weight和bias
* gradients：在backward时被计算出来，和parameters共同构成计算图中leaf node占据的显存
* optimizer state: 在optimizer中针对每个parameter维护的一些状态（momentum, variance）
* activation memory: 存储在每个tensor内部的计算结果（包括forward和backward的结果），构成计算图中其他节点的显存

静态：parameters, gradient, optimizer state，一些model中不在计算图中的常量

动态（每个epoch结束以后被释放）：optimizer state







