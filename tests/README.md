# MiniBrainGd 纯 C++ 单元测试

这些测试直接编译 MiniBrainGd 的 C++ 逻辑，不启动 Godot，也不加载
`MiniBrainGd.gdextension`。当前覆盖：

- `StatePooling<Scalar>` 和 `StatePooling<AutoDiffVar>` 的均值池化、最大值池化及 batch 隔离行为；
- Embedding 共享权重在多实体 backward 中的梯度累加；
- 按 `AIAgent` 训练结构组装的 Embedding、ReLU、Attention、StatePooling、GRU、Actor Heads 和 Critic；
- 随机输入的完整 forward；
- 与 `AIAgent` 相同形式的 PPO actor loss 和 MSE critic loss；
- 连续两轮 backward 和 Adam update。

训练冒烟测试使用一组较小且维度相容的参数（三个实体、`gru_hidden_dim ==
2 * embedding_dim`），并逐层初始化参数，目标是隔离验证 forward/loss/backward
计算链；它不等同于对 `AIAgent::Init` 默认参数组合的集成验证。

## Windows

前置条件：Python、SCons，以及 llvm-mingw。运行：

```bat
build_tests.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin
```

第二个参数可以指定并行任务数：

```bat
build_tests.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin 8
```

脚本会执行两步：

```bat
scons -f SConstruct.tests llvm_bin="C:\Files\programs\llvm-mingw-ucrt-x86_64\bin"
tests\bin\minibrain_gd_tests.exe
```

## Linux

前置条件：SCons 和 PATH 中可用的 C++17 编译器。运行：

```bash
./build_tests.sh
```

也可以传入并行任务数：

```bash
./build_tests.sh 8
```

## 与 GDExtension 构建的关系

测试使用独立的 `SConstruct.tests`，因此：

- 不编译 `godot-cpp`；
- 不链接 Godot；
- 不依赖 Godot 编辑器；
- 测试失败时直接返回非零退出码，适合放入 CI；
- 中间文件和测试程序分别写入 `tests/build/`、`tests/bin/`，不会覆盖扩展构建使用的 `src/*.o`；这些产物也会被现有忽略规则排除。

增加新的纯 C++ 测试时，在 `SConstruct.tests` 的 `test_objects` 中为测试源文件
增加一个 `env.Object(...)`。若测试某个 `src/*.cpp` 中的显式模板实例，也要为
对应实现文件增加独立的 `env.Object(...)`。
