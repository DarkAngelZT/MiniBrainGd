#!/usr/bin/env python

# 依次加载 godot-cpp 和 MiniMind，各依赖库自行封装构建细节。
env = SConscript("godot-cpp/SConstruct")
minimind_build = SConscript("MiniMind/SConscript", exports={"env": env})

env.Append(CPPPATH=[
    "src",
    "godot-cpp/include",
    "godot-cpp/gen/include",
    "godot-cpp/include/godot_cpp/core",
    ".",
])

sources = Glob("src/*.cpp")

# GDExtension 清单使用稳定文件名，不继承 godot-cpp 的“.dev”内部标记。
extension_suffix = env["suffix"].replace(".dev", "")

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "bin/libMiniBrainGd.{}.{}.framework/libMiniBrainGd.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
elif env["platform"] == "ios":
    if env["ios_simulator"]:
        library = env.StaticLibrary(
            "bin/libMiniBrainGd.{}.{}.simulator.a".format(
                env["platform"], env["target"]
            ),
            source=sources,
        )
    else:
        library = env.StaticLibrary(
            "bin/libMiniBrainGd.{}.{}.a".format(env["platform"], env["target"]),
            source=sources,
        )
else:
    library = env.SharedLibrary(
        "bin/libMiniBrainGd{}{}".format(extension_suffix, env["SHLIBSUFFIX"]),
        source=sources,
    )

# 插件链接等待 MiniMind 入口返回的构建节点。
env.Depends(library, minimind_build)

Default(library)
