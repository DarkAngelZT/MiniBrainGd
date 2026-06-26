#!/usr/bin/env python
import os
import sys

# You can find documentation for SCons and SConstruct files at:
# https://scons.org/documentation.html

# This lets SCons know that we're using godot-cpp, from the godot-cpp folder.
env = SConscript("godot-cpp/SConstruct")

# Configure include search paths to match VS Code's c_cpp_properties.json, using relative workspace paths.
# LLVM include paths are not needed here, only project headers.
env.Append(CPPPATH=[
    "src",
    "MiniBrain/Source",
    "MiniBrain/Source/ThirdParty",
    "godot-cpp/include",
    "godot-cpp/gen/include",
    "godot-cpp/include/godot_cpp/core",
    ".",
])

# Collects all .cpp files in the 'src' folder as compile targets.
sources = Glob("src/*.cpp")

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
            "bin/libMiniBrainGd.{}.{}.simulator.a".format(env["platform"], env["target"]),
            source=sources,
        )
    else:
        library = env.StaticLibrary(
            "bin/libMiniBrainGd.{}.{}.a".format(env["platform"], env["target"]),
            source=sources,
        )
else:
    library = env.SharedLibrary(
        "bin/libMiniBrainGd{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

# Selects the shared library as the default target.
Default(library)
