-------------------- Windows

1.Build Third Party projects with Release build (change it in Visual Studio). MAke sure to change CMAKE_C_FLAGS_RELEASE to /MD /Od /Ob0 /D NDEBUG otherwise toy will get CVODE error. Do not change
CMAKE_BUILD_TYPE if compiling using Visual Studio you will do it from Visual Studio interface

Build ALL_BUILD target in Visual Studio (it takes a while)
Build INSTALL target in Visual Studio


For more information about CVODE error see https://code.google.com/p/roadrunnerlib/wiki/VisualStudioBuild




2. Build RoadRunner project
a) Turn on BUILD_CC3D_EXTENSION in CMake GUI
b) Set CC3D_INSTALL_PREFIX CompuCell3D installation directory
c) Optional Turn off INSTALL_CC3D_RR_PYTHON_EXAMPLE


Build ALL_BUILD target in Visual Studio 
Build INSTALL target in Visual Studio


CVODE Error
When generating Visual Studio build files, the CMAKE_BUILD_TYPE have no effect, i.e. changing the Debug to Release will not cause the build to be a Release build. Instead, Debug and Release is manually set in the Visual Studio IDE. This do not mean, however, that one should use one set of build files for both Debug and Release. The problem in doing so, is that this would cause Debug and Release libraries to be generated and installed into the same output folder, which is not considered a good idea.

    The CMAKE_C_FLAGS_RELEASE 

    If you are building a release version of the ThirdParty libraries, the CMAKE_C_FLAGS_RELEASE flag need to be modified from its default. The default 

/MD /O2 /Ob2 /D NDEBUG

needs to be changed to

/MD /Od /Ob0 /D NDEBUG

The problem seem to be in the CVODE integration library, not being able to converge properly when full optimization is turned on. 

-------------------- Linux
------32 bit release
On 32 bit Ubuntu when compiling in the Release mode change CMAKE_C_FLAGS_RELEASE to -O0 -NDEBUG in ThirdParty Project

------64 bit release
No changes necesary RR works out of the box

-------------------- Apple OSX
not tested
