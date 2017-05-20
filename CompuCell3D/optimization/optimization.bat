@ECHO OFF
@SET PREFIX_CC3D=C:\CompuCell3D-64bit
REM @SET PYTHON_INSTALL_PATH=C:\CompuCell3D-64bit\Python27
@SET PYTHON_INSTALL_PATH=C:\Miniconda64\envs\cc3d_2015
@SET CC3D_RUN_SCRIPT=%PREFIX_CC3D%\runScript.bat

@SET OPTIMIZATIION_PYTHON_SCRIPT="d:\CC3D_GIT\optimization\optimization.py"

@set CURRENT_DIRECTORY=%CD%

cd %PREFIX_CC3D%

@SET exit_code=0

"%PYTHON_INSTALL_PATH%\python" %OPTIMIZATIION_PYTHON_SCRIPT% %* --cc3d-run-script=%CC3D_RUN_SCRIPT% --clean-workdirs 

REM --currentDir="%CURRENT_DIRECTORY%"

@SET exit_code= %errorlevel%

goto simulationend

:simulationend
   echo "SIMULATION FINISHED"
   cd %CURRENT_DIRECTORY%
   
echo "THIS IS EXIT CODE %exit_code%"   
exit /b %exit_code%    