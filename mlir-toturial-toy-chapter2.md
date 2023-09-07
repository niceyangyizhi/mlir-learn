# Emitting Basic MLIR

## Introduction

像LLVM这样的传统编译器，提供了一组固定的预定义类型和指令（通常是低级的/RISC风格的）。对于一个语言的编译器前端，它需要执行语言特定的类型检查、分析和转换，然后再生成LLVM IR。比如，clang使用AST既做静态分析，也做转换，例如通过AST节点复制和重写来进行C++模板实例化。这样就导致了许多编译器前端不得不反复实现编译流程中的重要组件，以便能做分析和转换。MLIR通过设计得具有可扩展性来解决这个问题。因此，在MLIR中仅有很少的指令（操作）和类型。

## Interfacing with MLIR

MLIR被设计成完全可扩展的基础设施：没有封闭的属性集、操作或类型。MLIR使用`Dialects`的概念来支持扩展性，MLIR为独立命名空间下的抽象提供了一个分组机制。

在MLIR中，`Operations`是抽象和计算的核心单元，在很多地方类似LLVM中的指令。`Operations`可以具有某个特定应用程序的语义，并且可以表示LLVM IR中的所有核心数据结构：指令、全局变量（如函数）、模块等。

这是Toy `transpose` 操作的汇编代码：

```
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

上面代码各部分的说明：
- `%t_tensor`: 操作产生的结果，名字前加了一个前缀以避免重名。一个操作可以产生零个或多个结果，每个结果都是SSA值，在Toy中，我们限定结果只能有一个。
- `"toy.transpose"`: 操作的名字，带有一个dialect命名空间前缀。可以解释为在Toy命名空间下的transpose操作。
- `(%tensor)`: 操作数列表（函数参数）
- `{ inplace = true }`: 零个或多个特殊的常量属性值。这里定义了一个`inplace`属性，值为`true`。
- `(tensor<2x3xf64>) -> tensor<3x2xf64>`: 参数类型和返回值类型。
- `loc("example/file/path":12:1)`: 此操作在源代码中的位置。

上面展示了一个操作的一般形式。在MLIR中操作是可扩展的。操作使用一组概念来描述，这些概念如下：
- 操作的名字
- 操作数列表
- 属性列表
- 操作结果类型列表
- 在源代码中的位置（用于调试）
- 后继块列表（主要用于分支）
- 区域列表（用于像函数这样的结构化操作）

MLIR旨在运行开发者自定义所有的IR元素，包括属性、操作、类型。同样，也可以将IR元素简化成上面的那些基本概念。这允许MLIR解析、比哦啊是和遍历任何操作的IR。我们可以把上面Toy操作放进一个`.mlir`文件中，然后使用*mlir-opt*来遍历它：

```
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```
这里的操作、类型和属性没有在MLIR中注册，因此是不安全的，下面是一个无效mlir的例子：

```
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```
这个IR的问题有：
- `"toy.print"`操作不是终止符
- 它应该有一个输入操作数，并且不该有返回值

## Defining a Toy Dialect

`Dialect`描述了一个语言的结构，并且提供了一个简便的方式来做高层次分析和转换。定义Dialect有两种方式：1. 使用C++定义，2. 使用ODS（Operation Definition Specification）框架以声明的方式定义，保存在`.td`文件中，会通过`tablegen`工具自动生成等价的C++定义。因为OSD定义更加清晰简洁，也容易生成说明文档，所以推荐使用ODS定义。

Toy Dialect的C++定义如下：

```
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

ODS声明如下：

```
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

可以运行如下命令，将上述ODS声明生成C++定义；

```
${build_root}/bin/mlir-tblgen -gen-dialect-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

定义完Toy dialect后，我们可以把它加载进MLIRContext：

```
context.loadDialect<ToyDialect>();
````

MLIRContext默认只加载MLIR内置的dialect（提供了一些核心IR组件），因此，对于自定义dialect，我们需要显式地加载。

## Defining Toy Operations

接着我们开始定义操作。这里我们定义一个`toy.constant`操作，它在Toy语言中表示一个常量。`toy.constant`在MLIR中是下面这样的：

```
%4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

`toy.constant`操作不需要操作数，接受一个名为“value”的dense element属性来表示常量值，返回一个RankedTensorType类型的结果。这个类继承了CRTP mlir::Op类。C++ CRTP是一种模板元编程技术，将派生类作为模板参数传递给基类模板，用于实现静态多态性，比普通的基于虚函数的动态多态性效率要更高。mlir::Op类还可以带一些traits自定义操作的行为。下面是`toy.constant`操作的C++定义：

```
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// We also provide a utility `getType` accessor that
                     /// returns the TensorType of the single result.
                     mlir::OpTraits::OneTypedResult<TensorType>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system. The name
  /// provided here must be prefixed by the parent dialect namespace followed
  /// by a `.`.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations may provide additional verification beyond what the attached
  /// traits provide.  Here we will ensure that the specific invariants of the
  /// constant operation are upheld, for example the result type must be
  /// of TensorType and matches the type of the constant `value`.
  LogicalResult verifyInvariants();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the `builder` classes to allow for easily
  /// generating instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

然后将这个操作注册进Toy Dialect中；

```
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
``` ¶

在MLIR中，跟操作相关的类有两个：`Operation`和`Op`，`Operation`描述了所有操作的通用行为，`Op`则表述了一个操作的特殊行为。在上面中我们通过继承`Op`定义了ConstantOp这一特别的操作。给出一个`Operation*`实例，我们可以把它转换成特定类型的`Op`，例如：

```
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

定义操作除了使用C++定义外，还可以使用ODS声明定义。ODS声明会在编译时自动扩展成一个等价的C++定义。ODS定义操作需要继承`Op`类，为了方便，我们为Toy dialect中的所有操作定义一个公共的基类。

```
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

接下来，我们通过继承`Toy_Op`类来定义ConstantOp：

```
def ConstantOp : Toy_Op<"constant"> {
}
```

运行下面的命令可以根据上述ODS声明生成对应的C++描述：

```
${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

然后，我们为ODS声明的操作添加输入参数和操作结果声明：

```
def ConstantOp : Toy_Op<"constant"> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

为参数或结果添加一个名字，比如`$value`，ODS会自动地生成一个对应的访问函数：`DenseElementsAttr ConstantOp::value()`。

下面，我们为操作添加一些描述信息。`summary`和`description`字段可以分别为操作添加概要信息和详细描述信息，这些信息可以自动生成markdown文档。

```
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

到这里，我们已经完成了C++定义中的大部分内容，下一步是定义验证器（verifier）。就像名称访问器（named accessor）一样，ODS框架会根据我们提供的约束，自动生成大量必要的验证逻辑。在大多数情况下，我们并不需要添加额外的验证逻辑。如果要手动添加额外的验证逻辑，那么操作需要重写`verifier`字段，`verifier`字段可以插入C++代码段，这些代码会作为`ConstantOp::verify`的一部分运行。

```
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Setting this bit
  // to `1` will generate a `::mlir::LogicalResult verify()` declaration on the
  // operation class that is called after ODS constructs have been verified, for
  // example the types of arguments and results. We implement additional verification
  // in the definition of this `verify` method in the C++ source file. 
  let hasVerifier = 1;
}
```

ODS声明操作的最后一个部分是定义`build`方法。ODS会自动生成一些简单的build方法，我们自己需要在`builders`字段中添加剩下的方法。

```
def ConstantOp : Toy_Op<"constant"> {
  ...

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];
}
```

## 指定一个自定义汇编代码格式

现在，我们可以生成Toy IR了，例如下面的Toy代码：

```
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

生成的IR为：

```
module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  }) {sym_name = "main", type = () -> ()} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

上面的Toy操作都是按照通用汇编格式打印的，也就是第一章节中描述的格式。MLIR允许使用C++代码或ODS声明来自定义汇编格式，从而使得打印内容更加具有可读性。

这里我们以`toy.print`操作作为例子。目前的`toy.print`操作汇编格式有点冗长，可以去掉很多字符，下面给出一种简洁的形式：

```
toy.print %5 : tensor<*xf64> loc(...)
```


为了自定义汇编格式，我们可以通过C++代码重写hasCustomAssemblyFormat字段，也可以通过ODS声明的方式重写assemblyFormat字段，声明的写法会通过工具自动映射成C++写法。这里先展示C++写法。

```
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // Divert the printer and parser to `parse` and `print` methods on our operation,
  // to be implemented in the .cpp file. More details on these methods is shown below.
  let hasCustomAssemblyFormat = 1;
}
```

printer和parser的C++实现如下：

```
/// The 'OpAsmPrinter' class is a stream that will allows for formatting
/// strings, attributes, operands, types, etc.
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}
```

C++写法完成后，现在来看看声明式写法。声明式写法主要由3个部分组成：
- Directives：内建函数的类型，可以带上参数列表
- Literals；被``包围的关键字或标点符号
- Variables：操作上的实体，即参数、结果等。在PrintOp中，变量指`$input`

声明式写法如下：

```
/// Consider a stripped definition of `toy.print` here.
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // In the following format we have two directives, `attr-dict` and `type`.
  // These correspond to the attribute dictionary and the type of a given
  // variable represectively.
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

精简后的汇编格式如下：

```
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

目前为止，我们介绍了ODS框架定义操作的一些概念，还有很多概念为能介绍，比如区域（regions）和可变操作数（variadic operands），具体参见[完整规范](https://mlir.llvm.org/docs/DefiningDialects/Operations/)




















