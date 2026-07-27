# MiniBrainGd 构建说明

## Windows 扩展构建

默认构建 `template_debug`：

```bat
build_all.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin
```

脚本对 debug 目标显式传入：

```text
dev_build=yes optimize=debug debug_symbols=yes
```

这会保留断言，使用适合断点的优化级别并生成调试信息。单独构建
`godot-cpp` 时，`build_godot_cpp.bat` 使用相同参数。

发布构建不追加这些 debug 参数：

```bat
build_all.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin template_release 8
```

## Linux 扩展构建

```bash
./build_all.sh template_debug 8
./build_all.sh template_release 8
```

`build_all.sh` 和 `build_godot_cpp.sh` 同样只对 `template_debug` 追加上述
三个参数。

## 纯 C++ 单元测试

单元测试使用独立构建文件，不构建或链接 Godot：

```bat
build_tests.bat C:\Files\programs\llvm-mingw-ucrt-x86_64\bin 8
```

Linux：

```bash
./build_tests.sh 8
```

更详细的测试结构见 `tests/README.md`。
