echo "begining file copy"
echo %SRC_DIR%
echo %PREFIX%
echo $SRC_DIR
echo $PREFIX

cd conda-recipes-compucell3d
copy compucell3d.bat %PREFIX%\compucell3d.bat

rem xcopy %SRC_DIR%/conda-recipes-compucell3d/compucell3d.bat %PREFIX% /s /e

rem touch compucell3d.bat
rem echo python -m cc3d.player5 > compucell3d.bat


