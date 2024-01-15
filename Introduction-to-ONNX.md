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

## Supported Types

ONNX针对张量的数值计算做了优化。一个张量是一个多维数组。张量通过以下进行定义：

- 一个类型: 元素的类型，一个张量中所有元素类型相同
- 一个形状: 维度的一个数组，这个数组可以是空的，维度也可以是NULL
- 一个连续数组: 它表示所有的值。

这个定义中不包含`strides`或者基于一个已经存在的张量定义一个张量的视图。ONNX张量是一个没有`stride`的密集数组。

### Element Type

ONNX设计之初是为了帮助部署深度学习模型。这也是为什么一开始ONNX张量的元素类型被设计为`float`（32 bits）。现在的版本中支持所有常用类型。字典`TENSOR_TYPE_MAP`给出了ONNX元素类型和`numpy`之间的对应关系。

```
import re
from onnx import TensorProto

reg = re.compile('^[0-9A-Z_]+$')

values = {}
for att in sorted(dir(TensorProto)):
    if att in {'DESCRIPTOR'}:
        continue
    if reg.match(att):
        values[getattr(TensorProto, att)] = att
for i, att in sorted(values.items()):
    si = str(i)
    if len(si) == 1:
        si = " " + si
    print("%s: onnx.TensorProto.%s" % (si, att))
```

```
 0: onnx.TensorProto.UNDEFINED
 1: onnx.TensorProto.FLOAT
 2: onnx.TensorProto.UINT8
 3: onnx.TensorProto.INT8
 4: onnx.TensorProto.UINT16
 5: onnx.TensorProto.INT16
 6: onnx.TensorProto.INT32
 7: onnx.TensorProto.INT64
 8: onnx.TensorProto.STRING
 9: onnx.TensorProto.BOOL
10: onnx.TensorProto.FLOAT16
11: onnx.TensorProto.DOUBLE
12: onnx.TensorProto.UINT32
13: onnx.TensorProto.UINT64
14: onnx.TensorProto.COMPLEX64
15: onnx.TensorProto.COMPLEX128
16: onnx.TensorProto.BFLOAT16
17: onnx.TensorProto.FLOAT8E4M3FN
18: onnx.TensorProto.FLOAT8E4M3FNUZ
19: onnx.TensorProto.FLOAT8E5M2
20: onnx.TensorProto.FLOAT8E5M2FNUZ
```

ONNX是强类型语言，且不支持隐式类型转换。在ONNX中，无法将两个不同元素类型的张量/矩阵相加，尽管这在其他语言中是可以的。这也是为什么必须在图中插入显式类型转换。

### Sparse Tensor

稀疏张量很适合表示有大量空元素的数组。ONNX支持2D稀疏张量。`SparseTensorProto `类可以通过定义`dims`、`indices`(int64)、`values`来表示稀疏张量。

### Other types

除了张量和稀疏张量，ONNX还支持张量列表、张量映射，张量映射的列表（`SequenceProto`、`MapProto`）。这些很少使用。

## What is an opset version?

算子集与ONNX版本相对应。ONNX每个小版本号增加时，算子集的版本也会增加。每次版本增加都会更新算子或引入新算子。

```
import onnx
print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())
```

```
1.16.0  opset= 21
```

算子集信息同样会附加在每一个ONNX图上。这是一个全局信息。它定义了图中所有算子的版本。算子`Add`在版本6，7，13和14中更新了。如果图的算子集是15，这意味着算子`Add`遵循版本14中的规范。如果图中算子集版本是12，那么算子`Add`遵循版本7中的规范。一个图中的算子遵循离图的算子集版本最近的算子定义（小于等于当前图的算子集版本）。

一个图也许包括来自多个域的算子，比如`ai.onnx`和`ai.onnx.ml`。在这种情况下，图必须为每个域定义一个全局算子集。这就是说，算子集的作用域是同一个域下的所有算子。

## Subgraphs, tests and loops

ONNX实现了分支和循环。它们都是把另一个ONNX图作为子图。这些结构通常都很慢，而且比较复杂。最好避免使用它们。

### If

算子`If`根据条件判断结果，执行两个图中的一个。

```
If(condition) then
    execute this ONNX graph (`then_branch`)
else
    execute this ONNX graph (`else_branch`)
```

### Scan

算子`Scan`实现了一个固定迭代次数的循环。它在输入的行（或任意其他维度）上迭代，并将输入结果在同样的维度上合并，这个给出一个实现欧几里得距离的例子：

### Loop

算子`Loop`可以实现for循环和while循环。输入有两种不同的处理方式。第一种类似`Scan`，在第一个维度将输入拼接成一个张量，这要求不同迭代的输入有着兼容的形状。第二种是把不同迭代中的输出放在一个列表中。

## Extensibility

ONNX定义了一组标准的算子：`ONNX Operators`。尽管如此，ONNX也支持用户自己在内置域或新创建的域下定义自己的算子。每一个节点都有一个类型、一个名称、带有名称的输入和输出，以及属性。只要一个节点被上面约束描述了，那么就可以添加进任意的ONNX图中。

平方距离可以使用算子`Scan`实现。但是一个叫做`CDist`的算子被证明会更快，因此值得花费努力在运行时中实现它。

ONNX图中的算子是仅有描述，实现是在运行时中的。

## Functions


函数是扩展ONNX规范的一个方法。一些模型会用很多相同的算子块。这时就可以创建一个由ONNX算子组成的函数。一旦函数被定义，它表现得就像一个算子，有自己的输入、输入和属性。

使用函数有两个好处。第一，它可以减少代码函数和增加可阅读性。第二，ONNX运行时可以利用这个信息来使得推理更快。运行时可以对函数有专门的实现，不必依赖于其中的算子。

## Shape (and Type) Inference

知晓结果的形状对执行ONNX图并非是必要的，但是知道这个信息可以让推理跑得很快。比如，你有以下图；

```
Add(x, y) -> z
Abs(z) -> w
```

如果`x`和`y`有着相同的形状，那么`z`和`w`也会有相同的形状。知道这个，就可以复用分配给`z`的内存，原地计算出绝对值`w`。形状推理可以帮助运行时管理内存，因此是很有用的。

ONNX包在大多数情况下都能根据输入形状，计算出标准ONNX算子的输出结果形状。对于标准算子之外的自定义算子，无法做形状推理。

## Tools

`netron`可以很方便地可视化ONNX图。唯一的缺点是不能修改图的内容。`onnx2py.py`可以根据ONNX图生成等价的python文件。这样可以通过修改python文件来修改ONNX图。`zetane`可以加载ONNX模型，并且在模型被执行的时候展示中间结果。

# ONNX with Python

这节主要关注ONNX提供的几个用于创建ONNX图的python API函数。

## A simple example: a linear regression

线性回归是最简单的机器学习模型，可以用如下公式表示：`Y = X A + B`。我们可以把它视为三个变量的函数 `Y = f(X, A, B)`，进一步分解为 `y = Add(MatMul(X, A), B)`。这就是我们需要在ONNX中表示的形式。首先，需要用ONNX算子实现一个函数。ONNX是强类型语言。一个函数的输入输出的形状和类型都必须定义。也就是说，我们需要`make function`中的四个函数来构建图：

- `make_tensor_value_info`: 声明一个变量（输入或输入），给出它的形状和类型
- `make_node`: 创建一个节点，定义它的算子类型、输入和输出
- `make_graph`: 根据前两个函数生成的对象，创建一个ONNX图
- `make_model`: 将图和附加的元数据合并在一起

在整个创建过程中，我们需要为图的每一个节点的每一个输入/输出起一个名称。图的输入输出是ONNX对象。字符串用来引用中间结果。下面是一个例子：

```
# imports

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined

Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

一个空的形状（None）意味着任意形状。一个形状被定义为`[None, None]`则表示这个对象是一个二维张量，每个维度的长度是任意的。

## Serialization

ONNX是基于protobuf构建的。它添加了必要的信息来描述一个机器学习模型，大部分情况下，ONNX用来序列化和反序列化一个模型。

### Model Serialization

模型需要保存下来以便之后部署。ONNX底层使用protobuf。这样可以最小化模型占用的硬盘空间。ONNX中的每个对象都能使用`SerializeToString`方法序列化。这里有一个整个模型序列化的例子：

```
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# The serialization
with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# display
print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

图可以使用函数`load`来重新加载。

```
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

# display
print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

上面的输出看起来完全相同。任何模型都可以使用这种方式序列化，除非它们的大小超过2Gb。`protobuf`限制了对象要小于这个阈值。之后，将会展示如何突破这个限制。

### Data Serialization

张量的序列化通过如下所示：

```
import numpy
from onnx.numpy_helper import from_array

numpy_tensor = numpy.array([0, 1, 4, 5, 3], dtype=numpy.float32)
print(type(numpy_tensor))

onnx_tensor = from_array(numpy_tensor)
print(type(onnx_tensor))

serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)
```

```
<class 'numpy.ndarray'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
<class 'bytes'>
```

反序列化如下所示：

```
from onnx import TensorProto
from onnx.numpy_helper import to_array

with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read()
print(type(serialized_tensor))

onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

numpy_tensor = to_array(onnx_tensor)
print(numpy_tensor)
```

```
<class 'bytes'>
<class 'onnx.onnx_ml_pb2.TensorProto'>
[0. 1. 4. 5. 3.]
```

下面的数据都可以使用这种方式：

```
import onnx
import pprint
pprint.pprint([p for p in dir(onnx)
               if p.endswith('Proto') and p[0] != '_'])
```

```
['AttributeProto',
 'FunctionProto',
 'GraphProto',
 'MapProto',
 'ModelProto',
 'NodeProto',
 'OperatorProto',
 'OperatorSetIdProto',
 'OperatorSetProto',
 'OptionalProto',
 'SequenceProto',
 'SparseTensorProto',
 'StringStringEntryProto',
 'TensorProto',
 'TensorShapeProto',
 'TrainingInfoProto',
 'TypeProto',
 'ValueInfoProto']
```

上面的代码也可以简化成下面的形式：

```
from onnx import load_tensor_from_string

with open("saved_tensor.pb", "rb") as f:
    serialized = f.read()
proto = load_tensor_from_string(serialized)
print(type(proto))
```

```
<class 'onnx.onnx_ml_pb2.TensorProto'>
```

## Initializer, default value

先前的模型假设线性回归的系数也作为模型的输入。这样并不简便。按照ONNX规范，它们应该作为模型的一部分，以常量或者初始化器（initializer）的形式。下面的例子将上面模型中的输入`A`、`B`改成初始化器。该软件包实现了两个函数，用于在numpy数组和ONNX格式之间进行转换。

- `onnx.numpy_helper.to_array`: 将ONNX格式转换成numpy
- `onnx.numpy_helper.from_array`: 将numpy格式转换成ONNX

```
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "X"
    input: "A"
    output: "AX"
    op_type: "MatMul"
  }
  node {
    input: "AX"
    input: "C"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  initializer {
    dims: 2
    data_type: 1
    name: "A"
    raw_data: "\000\000\000?\232\231\031\277"
  }
  initializer {
    dims: 1
    data_type: 1
    name: "C"
    raw_data: "\315\314\314>"
  }
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

同样，我们也可以遍历ONNX结构来检查初始化器是长什么样子的。

```
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# initializers
value = numpy.array([0.5, -0.6], dtype=numpy.float32)
A = numpy_helper.from_array(value, name='A')

value = numpy.array([0.4], dtype=numpy.float32)
C = numpy_helper.from_array(value, name='C')

# the part which does not change
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
node1 = make_node('MatMul', ['X', 'A'], ['AX'])
node2 = make_node('Add', ['AX', 'C'], ['Y'])
graph = make_graph([node1, node2], 'lr', [X], [Y], [A, C])
onnx_model = make_model(graph)
check_model(onnx_model)

print('** initializer **')
for init in onnx_model.graph.initializer:
    print(init)
```

在带有初始化器的模型中，只剩下了一个输入。输入`A`和`B`被移除了，它们也是可以保留的。在这种情况下，它们是可选的。每个和输入同名的初始化器被视作输入的默认值，如果没有模型没有传入输入的话，那么就会使用默认值。

## Attributes

一些算子需要属性，比如`Transpose`。让我们为表达式`y = XA' + B`构建一个ONNX图，等价于`y = Add(MatMul(X, Transpose(A)) + B)`。转置算子需要一个属性来表示在哪两个轴上进行转置: `perm=[1, 0]`。在`make_node`函数中可以为其添加一个带有名称的属性。

```
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# unchanged
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# added
node_transpose = make_node('Transpose', ['A'], ['tA'], perm=[1, 0])

# unchanged except A is replaced by tA
node1 = make_node('MatMul', ['X', 'tA'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# node_transpose is added to the list
graph = make_graph([node_transpose, node1, node2],
                   'lr', [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "A"
    output: "tA"
    op_type: "Transpose"
    attribute {
      name: "perm"
      ints: 1
      ints: 0
      type: INTS
    }
  }
  node {
    input: "X"
    input: "tA"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
```

*make*函数的完整列表如下所示：

```
import onnx
import pprint
pprint.pprint([k for k in dir(onnx.helper)
               if k.startswith('make')])
```

```
['make_attribute',
 'make_attribute_ref',
 'make_empty_tensor_value_info',
 'make_function',
 'make_graph',
 'make_map',
 'make_map_type_proto',
 'make_model',
 'make_model_gen_version',
 'make_node',
 'make_operatorsetid',
 'make_opsetid',
 'make_optional',
 'make_optional_type_proto',
 'make_sequence',
 'make_sequence_type_proto',
 'make_sparse_tensor',
 'make_sparse_tensor_type_proto',
 'make_sparse_tensor_value_info',
 'make_tensor',
 'make_tensor_sequence_value_info',
 'make_tensor_type_proto',
 'make_tensor_value_info',
 'make_training_info',
 'make_value_info']
```

## Opset and metadata

让我们加载之前保存的ONNX文件，然后看看它有哪些元数据。

```
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

for field in ['doc_string', 'domain', 'functions',
              'ir_version', 'metadata_props', 'model_version',
              'opset_import', 'producer_name', 'producer_version',
              'training_info']:
    print(field, getattr(onnx_model, field))
```

```
doc_string 
domain 
functions []
ir_version 9
metadata_props []
model_version 0
opset_import [version: 21
]
producer_name 
producer_version 
training_info []
```

这里大部分内容是空的，因为它们没有添加到我们创建的ONNX图中。其中有两项是有值的：

```
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

print("ir_version:", onnx_model.ir_version)
for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```
ir_version: 9
opset domain='' version=21
```

`IR`定义了ONNX语言的版本。`opset`定义了使用的算子版本。如果没有指明，ONNX默认使用已安装的包中最新的版本。也可以使用别的版本。

```
from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 14

for opset in onnx_model.opset_import:
    print("opset domain=%r version=%r" % (opset.domain, opset.version))
```

```
opset domain='' version=14
```

可以使用任意球遵循ONNX规范的算子集。在版本5中的`Reshape`算子把形状作为一个输入，而在版本1中，形状是作为一个属性。算子集表明了要遵循的ONNX规范。

其他的元数据可以保存任意信息、模型的生成方式、模型的不同版本号。

```
from onnx import load, helper

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

onnx_model.model_version = 15
onnx_model.producer_name = "something"
onnx_model.producer_version = "some other thing"
onnx_model.doc_string = "documentation about this model"
prop = onnx_model.metadata_props

data = dict(key1="value1", key2="value2")
helper.set_model_props(onnx_model, data)

print(onnx_model)
```

```
ir_version: 9
producer_name: "something"
producer_version: "some other thing"
model_version: 15
doc_string: "documentation about this model"
graph {
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  name: "lr"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  version: 21
}
metadata_props {
  key: "key1"
  value: "value1"
}
metadata_props {
  key: "key2"
  value: "value2"
}
```

字段`training_info`可以用来保存额外的图。

## Subgraph: test and loops

这些通常被归类为控制流。最好避免使用这些操作，因为它们没有矩阵操作的执行效率高。

### If

可以使用算子`If`实现分支判断。它会根据一个布尔值来决定执行两个子图中的一个。它不经常使用，因为一个函数通常需要一个batch中大量的比较结果。下面的例子计算出一个矩阵中所有浮点数的和，总和大于0就返回1，否则返回-1。

```
import numpy
import onnx
from onnx.helper import (
    make_node, make_graph, make_model, make_tensor_value_info)
from onnx.numpy_helper import from_array
from onnx.checker import check_model
from onnxruntime import InferenceSession

# initializers
value = numpy.array([0], dtype=numpy.float32)
zero = from_array(value, name='zero')

# Same as before, X is the input, Y is the output.
X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])

# The node building the condition. The first one
# sum over all axes.
rsum = make_node('ReduceSum', ['X'], ['rsum'])
# The second compares the result to 0.
cond = make_node('Greater', ['rsum', 'zero'], ['cond'])

# Builds the graph is the condition is True.
# Input for then
then_out = make_tensor_value_info(
    'then_out', onnx.TensorProto.FLOAT, None)
# The constant to return.
then_cst = from_array(numpy.array([1]).astype(numpy.float32))

# The only node.
then_const_node = make_node(
    'Constant', inputs=[],
    outputs=['then_out'],
    value=then_cst, name='cst1')

# And the graph wrapping these elements.
then_body = make_graph(
    [then_const_node], 'then_body', [], [then_out])

# Same process for the else branch.
else_out = make_tensor_value_info(
    'else_out', onnx.TensorProto.FLOAT, [5])
else_cst = from_array(numpy.array([-1]).astype(numpy.float32))

else_const_node = make_node(
    'Constant', inputs=[],
    outputs=['else_out'],
    value=else_cst, name='cst2')

else_body = make_graph(
    [else_const_node], 'else_body',
    [], [else_out])

# Finally the node If taking both graphs as attributes.
if_node = onnx.helper.make_node(
    'If', ['cond'], ['Y'],
    then_branch=then_body,
    else_branch=else_body)

# The final graph.
graph = make_graph([rsum, cond, if_node], 'if', [X], [Y], [zero])
onnx_model = make_model(graph)
check_model(onnx_model)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Save.
with open("onnx_if_sign.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# Let's see the output.
sess = InferenceSession(onnx_model.SerializeToString(),
                        providers=["CPUExecutionProvider"])

x = numpy.ones((3, 2), dtype=numpy.float32)
res = sess.run(None, {'X': x})

# It works.
print("result", res)
print()

# Some display.
print(onnx_model)
```

```
result [array([1.], dtype=float32)]

ir_version: 8
graph {
  node {
    input: "X"
    output: "rsum"
    op_type: "ReduceSum"
  }
  node {
    input: "rsum"
    input: "zero"
    output: "cond"
    op_type: "Greater"
  }
  node {
    input: "cond"
    output: "Y"
    op_type: "If"
    attribute {
      name: "else_branch"
      g {
        node {
          output: "else_out"
          name: "cst2"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 1
              data_type: 1
              raw_data: "\000\000\200\277"
            }
            type: TENSOR
          }
        }
        name: "else_body"
        output {
          name: "else_out"
          type {
            tensor_type {
              elem_type: 1
              shape {
                dim {
                  dim_value: 5
                }
              }
            }
          }
        }
      }
      type: GRAPH
    }
    attribute {
      name: "then_branch"
      g {
        node {
          output: "then_out"
          name: "cst1"
          op_type: "Constant"
          attribute {
            name: "value"
            t {
              dims: 1
              data_type: 1
              raw_data: "\000\000\200?"
            }
            type: TENSOR
          }
        }
        name: "then_body"
        output {
          name: "then_out"
          type {
            tensor_type {
              elem_type: 1
            }
          }
        }
      }
      type: GRAPH
    }
  }
  name: "if"
  initializer {
    dims: 1
    data_type: 1
    name: "zero"
    raw_data: "\000\000\000\000"
  }
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 15
}
```
这里`else`和`then`分支都很简单。`If`节点甚至可以用更快的`Where`节点替换。

### Scan

`Scan`的规范看起来很复杂。它适用于在张量的一个维度上进行迭代，并将结果存储在预先分配的张量中。

下面的例子实现了回归的问题一个经典最近邻居算法。第一步时计算输入特征X和训练集W的平方距离。接着是一个`TopK`算子来提取前K个最近的邻居。

```

```

## Functions

在构建模型的时候，函数可以用来减少代码的长度，也可以在运行时中通过提供优化的实现来加速推理。`make_function`用来定义一个函数。

### A function with no attribute

这里是一个简单的例子。函数的每一个输入都是在运行时才获知的对象。

```
import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor,
    make_graph, make_tensor_value_info, make_opsetid,
    make_function)
from onnx.checker import check_model

new_domain = 'custom'
opset_imports = [make_opsetid("", 14), make_opsetid(new_domain, 1)]

# Let's define a function for a linear regression

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

linear_regression = make_function(
    new_domain,            # domain name
    'LinearRegression',     # function name
    ['X', 'A', 'B'],        # input names
    ['Y'],                  # output names
    [node1, node2],         # nodes
    opset_imports,          # opsets
    [])                     # attribute names

# Let's use it in a graph.

X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

graph = make_graph(
    [make_node('LinearRegression', ['X', 'A', 'B'], ['Y1'], domain=new_domain),
     make_node('Abs', ['Y1'], ['Y'])],
    'example',
    [X, A, B], [Y])

onnx_model = make_model(
    graph, opset_imports=opset_imports,
    functions=[linear_regression])  # functions to add)
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
```

```
ir_version: 9
graph {
  node {
    input: "X"
    input: "A"
    input: "B"
    output: "Y1"
    op_type: "LinearRegression"
    domain: "custom"
  }
  node {
    input: "Y1"
    output: "Y"
    op_type: "Abs"
  }
  name: "example"
  input {
    name: "X"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "A"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  input {
    name: "B"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
          dim {
          }
        }
      }
    }
  }
  output {
    name: "Y"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 14
}
opset_import {
  domain: "custom"
  version: 1
}
functions {
  name: "LinearRegression"
  input: "X"
  input: "A"
  input: "B"
  output: "Y"
  node {
    input: "X"
    input: "A"
    output: "XA"
    op_type: "MatMul"
  }
  node {
    input: "XA"
    input: "B"
    output: "Y"
    op_type: "Add"
  }
  opset_import {
    domain: ""
    version: 14
  }
  opset_import {
    domain: "custom"
    version: 1
  }
  domain: "custom"
}
```

### A function with attributes

下面的函数与上面的例子基本是等价的，除了输入`B`被转换成了参数`bias`。
