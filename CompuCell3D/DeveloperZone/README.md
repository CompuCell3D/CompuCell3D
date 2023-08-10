When on OSX install MacOSX10.10.sdk into /opt/MacOSX10.10.sdk (for intel processors)
or MacOSX11.3.sdk into /opt/MacOSX11.3.sdk for Arm64 (M1, M2 etc) processors

The SDKs can be found at https://github.com/phracker/MacOSX-SDKs

When compiling and if you are in the conda environment make sure to
deactivate base environment

```commandline
cond deactivate
```

and then you can issue `make`, make install command