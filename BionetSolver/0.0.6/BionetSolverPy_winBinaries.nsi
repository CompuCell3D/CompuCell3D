
; BionetSolverPy_winBinaries.nsi 

;--------------------------------

; The name of the installer
Name "BionetSolverPy_winBinaries"

; The file to write
OutFile "BionetSolverPy_winBinaries.exe"

; The default installation directory
InstallDir $PROGRAMFILES32\CompuCell3D

; Request application privileges for Windows Vista and 7
RequestExecutionLevel highest
;;RequestExecutionLevel admin

;--------------------------------

; Pages

; Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------

Section "BionetSolverPy Components"

  ; Core binaries
  SetOutPath $INSTDIR\bin
  File install\bin\bionetsolver.dll
  File install\bin\soslib_shared.dll
  File install\bin\libsbml.dll
  File install\bin\xerces-c_2_7.dll
  File install\bin\bzip2.dll
  File install\bin\zlib1.dll
  
  ; Wrapper library files for bionetsolver
  SetOutPath $INSTDIR\lib\python
  File install\lib\python\BionetSolverPy.py
  File install\lib\python\_BionetSolverPy.pyd
  
  ; API for bionetsolver
  SetOutPath $INSTDIR\PythonSetupScripts
  File install\PythonSetupScripts\bionetAPI.py
  
  ; SBML models
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\sbmlModels
  File install\Demos\BionetSolverExamples\sbmlModels\MinimalDeltaNotch.xml
  File install\Demos\BionetSolverExamples\sbmlModels\CadherinCatenin_RamisConde2008.xml
  File install\Demos\BionetSolverExamples\sbmlModels\PK_BloodLiver.xml
  File install\Demos\BionetSolverExamples\sbmlModels\SimpleExample.xml
  
  ; MultiscaleSimulation example
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\MultiscaleSimulation
  File install\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation.xml
  File install\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation_Main.py
  File install\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation_Steppable.py
  
  ; OscillatingContactEnergies example
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies.xml
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies_Main.py
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies_Steppable.py
  
  ; OscillatingContactEnergiesFlex example
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex.xml
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex_Main.py
  File install\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex_Steppable.py
  
  ; DeltaNotchWithMitosis example
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\DeltaNotchWithMitosis
  File install\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis.xml
  File install\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis_Main.py
  File install\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis_Steppable.py
  
  ; TestCellDeath example
  SetOutPath $INSTDIR\Demos\BionetSolverExamples\TestCellDeath
  File install\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath.xml
  File install\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath_Main.py
  File install\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath_Steppable.py
  
  ; BionetSolverPy documentation
  SetOutPath $INSTDIR\Demos\BionetSolverExamples
  File doc\BionetSolver_QuickStartGuide.doc
  
  writeUninstaller $INSTDIR\BionetSolverPy_winBinaries_uninstall.exe

SectionEnd


Section "Uninstall"

  ; Remove the uninstaller first
  delete $INSTDIR\BionetSolverPy_winBinaries_uninstall.exe

  ; Remove core binaries
  Delete $INSTDIR\bin\bionetsolver.dll
  Delete $INSTDIR\bin\soslib_shared.dll
  Delete $INSTDIR\bin\libsbml.dll
  Delete $INSTDIR\bin\xerces-c_2_7.dll
  Delete $INSTDIR\bin\bzip2.dll
  Delete $INSTDIR\bin\zlib1.dll
  
  ; Remove Wrapper libraries for BionetSolverPy
  Delete $INSTDIR\lib\python\_BionetSolverPy.pyd
  Delete $INSTDIR\lib\python\BionetSolverPy.*
  
  ; Remove API for bionetsolver
  Delete $INSTDIR\PythonSetupScripts\bionetAPI.*
  
  ; Remove SBML models directory
  Delete $INSTDIR\Demos\BionetSolverExamples\sbmlModels\MinimalDeltaNotch.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\sbmlModels\CadherinCatenin_RamisConde2008.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\sbmlModels\PK_BloodLiver.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\sbmlModels\SimpleExample.xml
  RMDir $INSTDIR\Demos\BionetSolverExamples\sbmlModels
  
  ; MultiscaleSimulation example
  Delete $INSTDIR\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation_Main.py
  Delete $INSTDIR\Demos\BionetSolverExamples\MultiscaleSimulation\MultiscaleSimulation_Steppable.*
  RMDir $INSTDIR\Demos\BionetSolverExamples\MultiscaleSimulation
  
  ; OscillatingContactEnergies example
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies_Main.py
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergies_Steppable.*
  
  ; OscillatingContactEnergiesFlex example
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex_Main.py
  Delete $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies\OscillatingContactEnergiesFlex_Steppable.*
  RMDir $INSTDIR\Demos\BionetSolverExamples\OscillatingContactEnergies
  
  ; DeltaNotchWithMitosis example
  Delete $INSTDIR\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis_Main.py
  Delete $INSTDIR\Demos\BionetSolverExamples\DeltaNotchWithMitosis\DeltaNotchWithMitosis_Steppable.*
  RMDir $INSTDIR\Demos\BionetSolverExamples\DeltaNotchWithMitosis
  
  ; TestCellDeath example
  Delete $INSTDIR\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath.xml
  Delete $INSTDIR\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath_Main.py
  Delete $INSTDIR\Demos\BionetSolverExamples\TestCellDeath\TestCellDeath_Steppable.*
  RMDir $INSTDIR\Demos\BionetSolverExamples\TestCellDeath
  
  ; Remove BionetSolverPy documentation
  Delete $INSTDIR\Demos\BionetSolverExamples\BionetSolver_QuickStartGuide.doc
  
  RMDir $INSTDIR\Demos\BionetSolverExamples

SectionEnd