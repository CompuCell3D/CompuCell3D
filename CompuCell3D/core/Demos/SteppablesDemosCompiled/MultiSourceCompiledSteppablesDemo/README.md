# MultiSourceCompiledSteppablesDemo

This demo contains two independent C++ steppable targets:

- `Native_GrowthSteppable/GrowthClassSteppable.cpp` uses the nested `Utilities/Vector2D.h` and `Vector2D.cpp` files.
- `Native_SharedNumpyFieldsSteppable/SharedNumpyFieldsCompiledSteppable.cpp` uses the nested `Utilities/SharedFieldArrayUtils.h` and `SharedFieldArrayUtils.cpp` files.

The `.cc3d` file keeps one `Resource Type="Cpp"` entry per steppable. The target roots have different names, so each resource is discovered, compiled, and loaded as its own shared library. The extension source files are included automatically because they are below their respective target roots.
