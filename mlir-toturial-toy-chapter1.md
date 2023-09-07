# Toy Language and AST

## The Language

这是MLIR示例教程的第一章节，本教程将使用一个自定义的Toy语言作为教学的例子。Toy操作的数据是张量，出于简化的考虑，限制了张量的维数最大是2，只支持64位浮点类型（即C/C++中的`double`）。所有的值不需要进行类型声明，也不能改变，任何对数值的运算都会返回一个新的值。同样，也不用操心内存管理的问题，变量的内存会自动释放。

Toy程序示例：

```
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
```

添加一个自定义函数的示例程序：

```
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
```

## The AST

上面代码的AST如下：

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```

`examples/toy/Ch1/`目录中提供了Top的示例程序，可以通过以下命令生成AST：

```
path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast
```
由于本教程的重点不在于词法分析和语法分析，所以跳过了这两部分的实现，它们的实现可以在下面找到：
- `examples/toy/Ch1/include/toy/Lexer.h`
- `examples/toy/Ch1/include/toy/Parser.h`

词法分析和语法分析的实现过程可以参见LLVM的Kaleidoscope教程的前两章节，[Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html)





