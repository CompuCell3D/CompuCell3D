# Compiled Steppable ABI Demo

This demo mixes one compiled kernel ABI steppable with one ordinary Python steppable.

Build the native module before running the simulation.

## macOS / Clang

```bash
clang++ -O3 -std=c++17 -dynamiclib \
  Native/GrowthSteppable.cpp \
  -I../../../.. \
  -o Native/GrowthSteppable.dylib
```

## Linux / GCC

```bash
g++ -O3 -std=c++17 -fPIC -shared \
  Native/GrowthSteppable.cpp \
  -I../../../.. \
  -o Native/GrowthSteppable.so
```

## Windows / MSVC

```bat
cl /O2 /std:c++17 /LD Native\\GrowthSteppable.cpp /I..\\..\\..\\..
```

The compiled steppable updates `targetVolume` for type `A` cells every MCS. The Python steppable logs those
updated values every 10 MCS, verifying that the writes are visible immediately in the same scheduling pass.
