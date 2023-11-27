# ONNX Concept

ONNX可以视作一种专门用于数学函数的编程语言。它定义了机器学习模型实现推理函数所需要的全部操作。一个线性回归模型可以用如下方式表达：

```
def onnx_linear_regressor(X):
    "ONNX code for a linear regression"
    return onnx.Add(onnx.MatMul(X, coefficients), bias)
```

这与开发者在Python中写的表达式很类似。它同样能表示成一个图，一步步地展示如何将输入特征转化成预测。这也是为什么ONNX实现的机器学习模型常被称作ONNX图。

ONNX旨在提供一个同样的语言，任何机器学习框架都可以使用它来描述自己模型。第一种使用场景是，让在生产环境部署机器学习模型更加容易。可以为部署的环境实现一个专门的优化的ONNX解释器（运行时）。这样就可以构建一个统一的部署过程，与机器学习框架无关。ONNX提供了一个python运行时，可以用于执行ONNX模型和算子。这个运行时的目的是帮助理解ONNX的相关概念、调试ONNX的工具和转换器。ONNX运行时不要用于生产环境，它的性能不够好。

现在市面上有多种机器学习框架，每种框架的模型格式不一样，如果要为每一种框架写一个部署的运行时，那么将是一个沉重的负担。因此ONNX充当了学习框架和运行时之间的桥梁，每种学习框架把自己的模型转换成ONNX格式，那么只需要为ONNX写一个运行时就好了。

## Input, Output, Node, Initializer, Attributes

构建一个ONNX图是指用ONNX算子实现一个函数，一个线性回归模型可以用如下伪代码表示：

```
Input: float[M,K] x, float[K,N] a, float[N] c
Output: float[M, N] y

r = onnx.MatMul(x, a)
y = onnx.Add(r, c)
```

其中，`x`、`a`、 `c`是输入（Input），`r`是中间结果，`y`是输出（Output）。`MatMul`和`Add`是节点（Node），节点有输入输出和类型。

图也可以有初始化器（Initializer），当一个输入不会改变内容的时候，把它作为常量存储更有效率，比如线性回归的系数。

```
Input: float[M,K] x
Initializer: float[K,N] a, float[N] c
Output: float[M, N] xac

xa = onnx.MatMul(x, a)
xac = onnx.Add(xa, c)
```

属性（attribute）是节点（算子）的固定参数。算子`Gemm `有四个属性：alpha, beta, transA, transB。除非运行时允许通过API改变属性，否则图一旦加载，这些值不会再改变，在推理的时候也是保持不变。

## Serialization with protobuf

在生产环境部署一个机器学习模型通常需要把在训练模型时用到的依赖复制一份，大多数情况下使用docker。如果把模型转换成ONNX，那么生产环境只需要一个运行时来执行ONNX图就可以了。运行时可以用任意适合生产环境的语言开发，比如C、java、python、 javascript、C#、汇编、ARM等。

为了能实现上述目标，ONNX图需要被保存下来。ONNX使用protobuf来将图序列化成单个的块。这样做的目标是尽量减少模型的大小。

## Metadata

机器学习模型在持续更新中。记录模型的版本、作者、训练方式是很重要的。ONNX提供了方法来保存模型的这些信息。

- doc_string: 人类可阅读的关于此模型的文档，支持markdown。
- domain: 一个反转的DNS名称用于表明模型的命名空间或域，比如‘org.onnx’。
- metadata_props: 带名称的元数据，以字典的形式`map<string,string>`。
- model_author: 用逗号分隔的一组名字，表示模型的作者或组织。
- model_license: 证书的名称或URL。
- model_version: 模型的版本，整数。
- producer_name: 用于构建模型的工具名称。
- producer_version: 构建模型的工具的版本。
- training_info: 可选的扩展部分，用于表示模型的训练信息。

## List of available operators and domains

ONNX算子清单包括标准矩阵算子（Add, Sub, MatMul, Transpose, Greater, IsNaN, Shape, Reshape…）、聚合（ReduceSum, ReduceMin, …）、图像变换（Conv, MaxPool, …）、深度神经网络层（RNN, DropOut, …）、激活函数（Relu, Softmax, …）。清单囊括了实现机器学习推理函数的主要操作。ONNX没有实现每一个存在的机器学习算子，清单的大小是有限的。

清单上的大部分算子在**ai.onnx**域下，一个域由一组算子定义。其中，少量算子是专门用于处理文本的，但是它们很难满足需要。同样，这个域下也没有基于树的机器模型的算子。在另一个域`ai.onnx.ml`下，包含了基于树的模型（TreeEnsemble Regressor, …）、预处理（OneHotEncoder, LabelEncoder, …）、SVM模型（SVMRegressor, …）、填补器（Imputer）。

ONNX只定义了这两个域，但是支持自定义域和算子。