set BUILD_CONFIG=Release

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE:STRING="%BUILD_CONFIG%" ^
      -G "Ninja" ^
      -DCMAKE_PREFIX_PATH="%PREFIX%" ^
      -DCMAKE_FIND_ROOT_PATH="%LIBRARY_PREFIX%" ^
      -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
      -DNO_OPENCL:BOOLEAN=ON ^
      -DWINDOWS_DEPENDENCIES_INSTALL_ENABLE=OFF ^
      -DBUILD_STANDALONE=OFF ^
      -DPython_EXECUTABLE=%PYTHON% ^
      "%SRC_DIR%/CompuCell3D"
if errorlevel 1 exit 1

cmake --build . --config "%BUILD_CONFIG%" --target install
if errorlevel 1 exit 1