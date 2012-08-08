@ECHO OFF
@SET PREFIX_CC3D=D:\Program Files\COMPUCELL3D_3.4.05

set xmlFile=""
set pythonScript=""
set silentMode=""
set consoleMode=""
set scrDsc=""
set noXServerMode="false"
set outDir=""
set noOutputFlag=""




:CHECKFORSWITCHES
if '%1'=='-i' goto xml
if '%1'=='-p' goto python
if '%1'=='-s' goto screenshot
if '%1'=='-o' goto outputDirectory
if '%1'=='--silent' goto silent
if '%1'=='--noXServer' goto noXServer
if '%1'=='--noOutput' goto noOutput
if '%1'=='--console' goto console
if '%1'=='--help' goto usage
if '%1'=='-h' goto usage

goto begin



CLS

:xml
echo "got xml tag"
set xmlFile=%2
shift
shift
GOTO CHECKFORSWITCHES

:python
echo "got python tag"
set pythonScript=%2
shift
shift
GOTO CHECKFORSWITCHES

:screenshot
echo "got screenshot tag"
set scrDsc=%2
shift
shift
GOTO CHECKFORSWITCHES

:outputDirectory
echo "got outputDirectory tag"
set outDir=%2
shift
shift
GOTO CHECKFORSWITCHES

:silent
echo "got silentMode tag"
set silentMode=TRUE
shift
GOTO CHECKFORSWITCHES

:noXServer
echo "got noXServer tag"
set noXServerMode=TRUE
shift
GOTO CHECKFORSWITCHES

:noOutput
echo "got noOutput tag"
set noOutputFlag=TRUE
shift
GOTO CHECKFORSWITCHES

:console
echo "got console tag"
set consoleMode=TRUE
shift
GOTO CHECKFORSWITCHES

:usage
echo "HELP FUNCTION"
   echo "USAGE: ./compucell3d.sh -i <xmlFile> -p <pythonScript>"
   echo " -s <ScreenshotDescriptionFile> -o <outputDirectory>"
   echo " [--noOutput] [--silent] [--console] ."
   echo "--silent mode will use Player to save screenshots but will not display GUI."
   echo "You must provide screenshot description file to run in silent mode. "
   echo "--console mode will run basic CompuCell3D kernel but will only be able to take as an input xml file."
   echo "No screenshots are stored in console mode"
   echo "To use console mode effectively one would need to develop modules "
   echo "that would either save or display lattice snapshots."
   echo "For that reason it is recommended that you use CompuCellPlayer"
   echo "as it gives you nice set of visualization tools"

   echo "--noOutput mode will run CompuCell3D simulation but will not store any output."
goto simulationend

:begin
echo "DONE"

echo "XML FILE= %xmlFile%"
echo "pythonScript= %pythonScript%"
echo "scrDsc= %scrDsc%"
echo "outDir= %outDir%"



@SET PATH=%PREFIX_CC3D%\bin;%PATH%

@SET COMPUCELL3D_PLUGIN_PATH=%PREFIX_CC3D%\lib\CompuCell3DPlugins
@SET COMPUCELL3D_STEPPABLE_PATH=%PREFIX_CC3D%\lib\CompuCell3DSteppables
@SET SWIG_LIB_INSTALL_DIR=%PREFIX_CC3D%\lib\python
@SET PYTHON_MODULE_PATH=%PREFIX_CC3D%\pythonSetupScripts
@SET PYTHON_MODULE_EXTRA_PATH=%CD%

@SET PATH=c:\Program Files\VTK\bin;%PATH%
@SET PATH=%PREFIX_CC3D%\lib\extralibs;%PATH%

@SET PATH=%COMPUCELL3D_PLUGIN_PATH%;%PATH%
@SET PATH=%COMPUCELL3D_STEPPABLE_PATH%;%PATH%

python xml_python_expat_1.py
goto simulationend

:: ********************************************* fixing file paths *********************************************
@set CURRENT_DIRECTORY=%CD%

if not %xmlFile%=="" (
   if  exist "%CURRENT_DIRECTORY%\%xmlFile%" (SET xmlFile="%CURRENT_DIRECTORY%\%xmlFile%")
)
if not %pythonScript%=="" (
   if  exist "%CURRENT_DIRECTORY%\%pythonScript%" (SET pythonScript="%CURRENT_DIRECTORY%\%pythonScript%")
)
if not %scrDsc%=="" (
   if  exist "%CURRENT_DIRECTORY%\%scrDsc%" (SET scrDsc="%CURRENT_DIRECTORY%\%scrDsc%")
)

:: *************************************************************************************************************


cd %PREFIX_CC3D%

if not %consoleMode%=="" (
  if not %xmlFile%=="" (
     echo "GOT SIMULATION IN CONSOLE MODE"
     "%PREFIX_CC3D%\bin\CompuCell3D.exe" %xmlFile%
     goto simulationend
  )

  if %xmlFile%=="" (
     echo "YOU NEED TO PROVIDE XML FILE NAME"
     goto usage
  )

)



if not %xmlFile%=="" (
     if %silentMode%=="" (
           if %pythonScript%=="" (
              if %noOutputFlag%=="" (
                 if %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --outputDirectory=%outDir%
                        goto simulationend
                    )

                 )
                 if not %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --screenshotDescription=%scrDsc%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --screenshotDescription=%scrDsc% --outputDirectory=%outDir%
                        goto simulationend
                    )

                 )

              )
              if not %noOutputFlag%=="" (
                "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --noOutput
                goto simulationend
              )

           )
           if not %pythonScript%=="" (
              if %noOutputFlag%=="" (
                 if %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --outputDirectory=%outDir%
                        goto simulationend
                    )


                 )
                 if not %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --screenshotDescription=%scrDsc%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --screenshotDescription=%scrDsc% --outputDirectory=%outDir%
                        goto simulationend
                    )

                 )

              )
              if not %noOutputFlag%=="" (
                "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --noOutput
                goto simulationend
              )
           )
     )

   if not %silentMode%=="" (
       if %pythonScript%=="" (
          if %noOutputFlag%=="" (
             if %scrDsc%=="" (
                if %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --silent
                   goto simulationend
                )
                if not %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --silent --outputDirectory=%outDir%
                   goto simulationend
                )

             )
             if not %scrDsc%=="" (
                if %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --silent --screenshotDescription=%scrDsc%
                   goto simulationend
                )
                if not %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --silent --screenshotDescription=%scrDsc%  --outputDirectory=%outDir%
                   goto simulationend
                )

             )

          )
          if not %noOutputFlag%=="" (
            "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --silent --noOutput
            goto simulationend
          )

       )
       if not %pythonScript%=="" (
          if %noOutputFlag%=="" (
             if %scrDsc%=="" (
                if %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent
                   goto simulationend
                )
                if not %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --outputDirectory=%outDir%
                   goto simulationend
                )

             )
             if not %scrDsc%=="" (
                if %outDir%=="" (
                    "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --screenshotDescription=%scrDsc%
                    goto simulationend
                )
                if not %outDir%=="" (
                    "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --screenshotDescription=%scrDsc% --outputDirectory=%outDir%
                    goto simulationend
                )

             )

          )
          if not %noOutputFlag%=="" (
            "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --noOutput
            goto simulationend
          )

       )
   )


)


rem *************************** Python file supplied but no xml *********************************

if %xmlFile%=="" (
     if %silentMode%=="" (
           if not %pythonScript%=="" (
              if %noOutputFlag%=="" (
                 if %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --outputDirectory=%outDir%
                        goto simulationend
                    )


                 )
                 if not %scrDsc%=="" (
                    if %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --screenshotDescription=%scrDsc%
                        goto simulationend
                    )
                    if not %outDir%=="" (
                        "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --screenshotDescription=%scrDsc% --outputDirectory=%outDir%
                        goto simulationend
                    )

                 )

              )
              if not %noOutputFlag%=="" (
                "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --noOutput
                goto simulationend
              )
           )
     )

   if not %silentMode%=="" (
       if not %pythonScript%=="" (
          if %noOutputFlag%=="" (
             if %scrDsc%=="" (
                if %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent
                   goto simulationend
                )
                if not %outDir%=="" (
                   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --outputDirectory=%outDir%
                   goto simulationend
                )

             )
             if not %scrDsc%=="" (
                if %outDir%=="" (
                    "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --screenshotDescription=%scrDsc%
                    goto simulationend
                )
                if not %outDir%=="" (
                    "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --screenshotDescription=%scrDsc% --outputDirectory=%outDir%
                    goto simulationend
                )

             )

          )
          if not %noOutputFlag%=="" (
            "%PREFIX_CC3D%\bin\CompuCellPlayer.exe" --xml=%xmlFile% --pythonScript=%pythonScript% --silent --noOutput
            goto simulationend
          )

       )
   )


)

rem ******************************************

if %xmlFile%=="" (
   "%PREFIX_CC3D%\bin\CompuCellPlayer.exe"
   goto simulationend
)



:simulationend
   echo "SIMULATION FINISHED"
   cd %CURRENT_DIRECTORY%
