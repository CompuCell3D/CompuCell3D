############################################################################################
#      NSIS Installation Script created by NSIS Quick Setup Script Generator v1.09.18
#               Entirely Edited with NullSoft Scriptable Installation System                
#              by Vlasis K. Barkas aka Red Wine red_wine@freemail.gr Sep 2006               
############################################################################################

!define APP_NAME "CompuCell3D"
!define COMP_NAME "Biocomplexity Institute"
!define WEB_SITE "http://www.compucell3d.org"
!define VERSION "<VERSION>"
!define COPYRIGHT "Biocomplexity Institute © 2008"
!define DESCRIPTION "Application"
!define INSTALLER_NAME "<INSTALLER_NAME>"
!define MAIN_APP_EXE "compucell3d.bat"
!define INSTALL_TYPE "SetShellVarContext all"
!define REG_ROOT "HKLM"
!define REG_APP_PATH "Software\Microsoft\Windows\CurrentVersion\App Paths\${MAIN_APP_EXE}"
!define UNINSTALL_PATH "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

!define REG_START_MENU "Start Menu Folder"
### CUSTOM MODIFICATION
!include "FileFunc.nsh"
!define INSTALLATION_SOURCE_DIR "<INSTALLATION_SOURCE_DIR>"
!define CELLDRAW_EXE "celldraw.bat"
!define TWEDIT_EXE "twedit++.bat"


!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\cc3d_128x128_logo_setup.ico"

Var PYTHON_PATH27         ; holding the python path
Var numpyErrorCode        ; holding error code for numpy import attempt
Var win32guiErrorCode        ; holding error code for win32gui import attempt


Var CURRENT_UNINSTALLER         ; name of the current uninstaller

InstallDir ""

Function .onInit
; ${GetRoot} "$PROGRAMFILES" $SUGGESTED_INSTALL_PATH 
    ${GetRoot} "$PROGRAMFILES" $R0
    ; MessageBox MB_OK " Installation dir $R0"
    ; strcpy $SUGGESTED_INSTALL_PATH $R0
    
    StrCpy $InstDir "$R0\CompuCell3D"
    
FunctionEnd
### END OF CUSTOM MODIFICATION

var SM_Folder

######################################################################

VIProductVersion  "${VERSION}"
VIAddVersionKey "ProductName"  "${APP_NAME}"
VIAddVersionKey "CompanyName"  "${COMP_NAME}"
VIAddVersionKey "LegalCopyright"  "${COPYRIGHT}"
VIAddVersionKey "FileDescription"  "${DESCRIPTION}"
VIAddVersionKey "FileVersion"  "${VERSION}"

######################################################################

SetCompressor ZLIB
Name "${APP_NAME}"
Caption "${APP_NAME}"
OutFile "${INSTALLER_NAME}"
BrandingText "${APP_NAME}"
XPStyle on
InstallDirRegKey "${REG_ROOT}" "${REG_APP_PATH}" ""
InstallDir "$PROGRAMFILES\CompuCell3D"

######################################################################

!include "MUI.nsh"

!define MUI_ABORTWARNING
!define MUI_UNABORTWARNING

!insertmacro MUI_PAGE_WELCOME

!ifdef LICENSE_TXT
!insertmacro MUI_PAGE_LICENSE "${LICENSE_TXT}"
!endif

!insertmacro MUI_PAGE_DIRECTORY

!ifdef REG_START_MENU
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "CompuCell3D"
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "${REG_ROOT}"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${UNINSTALL_PATH}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "${REG_START_MENU}"
!insertmacro MUI_PAGE_STARTMENU Application $SM_Folder
!endif

!insertmacro MUI_PAGE_INSTFILES

!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM

!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

######################################################################
### CUSTOM MODIFICATION
Section -Uninstaller

    strcpy $CURRENT_UNINSTALLER ""    
    ReadRegStr $CURRENT_UNINSTALLER ${REG_ROOT} "${UNINSTALL_PATH}\" "UninstallString"
    strcmp $CURRENT_UNINSTALLER "" "" foundUninstaller
    goto notfoundUninstaller
    foundUninstaller:   
        ;MessageBox MB_OK "Exsisting copy of CompuCell3D will be uninstalled now.$\nPLEASE make sure to backup existing simulation"
        MessageBox MB_YESNO "Found existing CompuCell3D installation.$\nWould you like unistall it now (recommended)?$\n Before uninstalling PLEASE backup existing simulations" /SD IDYES IDNO NoUninstall
        
        ; ExecWait "$CURRENT_UNINSTALLER"
        ExecWait '"$CURRENT_UNINSTALLER" _?=$INSTDIR'
        MessageBox MB_OK " Uninstall Succesful "
        goto uninstallSectionComplete
        NoUninstall:
            MessageBox MB_OK " No uninstallation performed "
            goto uninstallSectionComplete
            
    notfoundUninstaller:
    uninstallSectionComplete:

SectionEnd


Section -PythonDependencies
    DetailPrint "Checking for Python 2.5"
    call CheckPython27
    
    SetOutPath $INSTDIR\Prerequisites
        File "${INSTALLATION_SOURCE_DIR}\Prerequisites\numpy-1.5.1-win32-superpack-python2.7.exe"               
        ExecWait '$PYTHON_PATH27python -c "import numpy"' $numpyErrorCode
        strcmp $numpyErrorCode "0" "" numpyNotInstalled
        
        goto numpyInstalled
        numpyNotInstalled:
            MessageBox MB_OK "  NUMPY  IS NOT INSTALLED. Starting numpy installer..."
            ExecWait "$INSTDIR\Prerequisites\numpy-1.5.1-win32-superpack-python2.7.exe"
            ; goto finishSection
        
        numpyInstalled:

    SetOutPath $INSTDIR\Prerequisites
        File "${INSTALLATION_SOURCE_DIR}\Prerequisites\pywin32-216.win32-py2.7.exe"               
        ExecWait '$PYTHON_PATH27python -c "import win32gui"' $win32guiErrorCode
        strcmp $win32guiErrorCode "0" "" pywin32NotInstalled
        
        goto pywin32Installed
        pywin32NotInstalled:
            MessageBox MB_OK "  PyWin32  IS NOT INSTALLED. Starting pywin32 installer..."
            ExecWait "$INSTDIR\Prerequisites\pywin32-216.win32-py2.7.exe"
            goto finishSection
        
        pywin32Installed:
        
        finishSection: 
    
SectionEnd

Section -Prerequisites

  SetOutPath $INSTDIR\Prerequisites
    File "${INSTALLATION_SOURCE_DIR}\Prerequisites\vcredist_x86.exe"
    ; ExecWait "$INSTDIR\Prerequisites\vcredist_x86.exe /q:a /c:$\"VCREDI~1.EXE /q:a /c:$\"$\"msiexec /i vcredist.msi /qb!$\"$\" $\""           
    ExecWait "$INSTDIR\Prerequisites\vcredist_x86.exe /q /norestart"
    
    
    Goto vs2008Libs
  vs2008Libs:
  
SectionEnd


### CUSTOM MODIFICATION
######################################################################


Section -MainProgram
${INSTALL_TYPE}
SetOverwrite ifnewer
<INSTALL_FILES>

### CUSTOM MODIFICATION 
# MessageBox MB_OK ' THIS IS Python Path  $PYTHON_PATH27python $INSTDIR\scriptSetup.py $INSTDIR'
DetailPrint "Postinstallation ..."
 ExecWait '$PYTHON_PATH27python "$INSTDIR\scriptSetup.py" "$INSTDIR" $PYTHON_PATH27 '
 #removing unnecessary files
 Delete "$INSTDIR\Prerequisites\vcredist_x86.exe"
 Delete "$INSTDIR\Prerequisites\numpy-1.5.1-win32-superpack-python2.7.exe"
 Delete "$INSTDIR\Prerequisites\pywin32-216.win32-py2.7.exe"
 Delete "$INSTDIR\Prerequisites\python-2.7.1.msi"

  
### END OF CUSTOM MODIFICATION 


SectionEnd

######################################################################
### CUSTOM MODIFICATION - Make sure to change uninstall.exe to uninstall-cc3d.exe everywhere in the installer script
Section -Icons_Reg

#!define CELLDRAW_DOCUMENTATION_LINK "file://$INSTDIR/CellDraw/CellDraw_manual/index.html"

SetOutPath "$INSTDIR"
WriteUninstaller "$INSTDIR\uninstall-cc3d.exe"

!ifdef REG_START_MENU
!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
CreateDirectory "$SMPROGRAMS\$SM_Folder"
CreateShortCut "$SMPROGRAMS\$SM_Folder\${APP_NAME}.lnk" "$INSTDIR\${MAIN_APP_EXE}" "CompuCell3D Run Script" "$INSTDIR\icons\cc3d_128x128_logo.ico"
CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${MAIN_APP_EXE}" "CompuCell3D Run Script" "$INSTDIR\icons\cc3d_128x128_logo.ico"
CreateShortCut "$SMPROGRAMS\$SM_Folder\CellDraw.lnk" "$INSTDIR\${CELLDRAW_EXE}" "Tool to prepare initial cell layout" "$INSTDIR\CellDraw\icons\CellDraw_64x64.ico"
CreateShortCut "$SMPROGRAMS\$SM_Folder\Twedit++.lnk" "$INSTDIR\${TWEDIT_EXE}" "" "$INSTDIR\Twedit++\icons\twedit-icon.ico"
CreateShortCut "$SMPROGRAMS\$SM_Folder\Uninstall ${APP_NAME}.lnk" "$INSTDIR\uninstall-cc3d.exe"

!ifdef WEB_SITE
; WriteIniStr "Website www.compucell3d.org" "InternetShortcut" "URL" "${WEB_SITE}"
; CreateShortCut "$SMPROGRAMS\$SM_Folder\${APP_NAME} Website.lnk" "Website www.compucell3d.org" "Website logo" "$INSTDIR\icons\cc3d_64x64_logo_www.ico"

WriteIniStr "$INSTDIR\${APP_NAME} website.url" "InternetShortcut" "URL" "${WEB_SITE}"
CreateShortCut "$SMPROGRAMS\$SM_Folder\${APP_NAME} Website.lnk" "$INSTDIR\${APP_NAME} website.url" "Website logo" "$INSTDIR\icons\cc3d_64x64_logo_www.ico"

; WriteIniStr "$INSTDIR\CellDraw_Documentation.url" "InternetShortcut" "URL" "${CELLDRAW_DOCUMENTATION_LINK}"
; CreateShortCut "$SMPROGRAMS\$SM_Folder\CellDraw Documentation.lnk" "$INSTDIR\CellDraw_Documentation.url" "CellDraw documentation logo" "$INSTDIR\CellDraw\icons\CellDraw_64x64_doc.ico"

!endif
!insertmacro MUI_STARTMENU_WRITE_END
!endif
WriteRegStr ${REG_ROOT} "${REG_APP_PATH}" "" "$INSTDIR\${MAIN_APP_EXE}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayName" "${APP_NAME}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "UninstallString" "$INSTDIR\uninstall-cc3d.exe"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayIcon" "$INSTDIR\${MAIN_APP_EXE}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayVersion" "${VERSION}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "Publisher" "${COMP_NAME}"

!ifdef WEB_SITE
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "URLInfoAbout" "${WEB_SITE}"
!endif
SectionEnd
### END OF CUSTOM MODIFICATION 
######################################################################

Section Uninstall
${INSTALL_TYPE}
<DELETE_FILES>
######################################################################
### CUSTOM MODIFICATION
Delete "$INSTDIR\uninstall-cc3d.exe"
!ifdef WEB_SITE
Delete "$INSTDIR\${APP_NAME} website.url"
; Delete "$INSTDIR\CellDraw_Documentation.url"
!endif

RmDir "$INSTDIR"

!ifdef REG_START_MENU
!insertmacro MUI_STARTMENU_GETFOLDER "Application" $SM_Folder
Delete "$SMPROGRAMS\$SM_Folder\${APP_NAME}.lnk"
Delete "$SMPROGRAMS\$SM_Folder\Uninstall ${APP_NAME}.lnk"
Delete "$SMPROGRAMS\$SM_Folder\CellDraw.lnk" 
Delete "$SMPROGRAMS\$SM_Folder\Twedit++.lnk"
!ifdef WEB_SITE
; Delete "$SMPROGRAMS\$SM_Folder\CellDraw Documentation.lnk"
Delete "$SMPROGRAMS\$SM_Folder\${APP_NAME} Website.lnk"

!endif
Delete "$DESKTOP\${APP_NAME}.lnk"

RmDir "$SMPROGRAMS\$SM_Folder"
!endif


DeleteRegKey ${REG_ROOT} "${REG_APP_PATH}"
DeleteRegKey ${REG_ROOT} "${UNINSTALL_PATH}"
SectionEnd

### END OF CUSTOM MODIFICATION
######################################################################

######################################################################
### CUSTOM MODIFICATION
Function CheckPython27

  strcpy $PYTHON_PATH27 ""
  ReadRegStr $PYTHON_PATH27 HKLM "SOFTWARE\Python\PythonCore\2.7\InstallPath" ""
  strcmp $PYTHON_PATH27 "" checkLocal found
  

checkLocal:
  strcpy $PYTHON_PATH27 ""
  ReadRegStr $PYTHON_PATH27 HKCU "SOFTWARE\Python\PythonCore\2.7\InstallPath" ""
  strcmp $PYTHON_PATH27 "" notFound found

found:
  ClearErrors  
  goto exit

    
notfound:
  #python probably not installed
    
    
    SetOutPath $INSTDIR\Prerequisites
        File "${INSTALLATION_SOURCE_DIR}\Prerequisites\python-2.7.1.msi"                   
        MessageBox MB_OK "  Python27  IS NOT INSTALLED. Starting the installer..."
        ; ExecWait "$INSTDIR\Prerequisites\python-2.7.1.msi"
        ExecWait '"msiexec" /i "$INSTDIR\Prerequisites\python-2.7.1.msi"'

        goto recheckPython27
    
  ; MessageBox MB_OK "Python 2.7 is required for CC3D but not present."
  ; Abort
recheckPython27:
  strcpy $PYTHON_PATH27 ""
  ReadRegStr $PYTHON_PATH27 HKLM "SOFTWARE\Python\PythonCore\2.7\InstallPath" ""
  strcmp $PYTHON_PATH27 "" checkLocalInstalled foundInstalled
  

checkLocalInstalled:
  strcpy $PYTHON_PATH27 ""
  ReadRegStr $PYTHON_PATH27 HKCU "SOFTWARE\Python\PythonCore\2.7\InstallPath" ""
  strcmp $PYTHON_PATH27 "" notFoundInstalled foundInstalled

foundInstalled:
  ClearErrors
  goto exit
  
notfoundInstalled:
  #python probably not installed

    
  MessageBox MB_OK "Python 2.7 still cannot be found. PLease install Python2.7 manuall and restart CC3D installer."
  Abort
  
  
  strcpy $PYTHON_PATH27 ""
exit:

FunctionEnd
### END OF CUSTOM MODIFICATION
