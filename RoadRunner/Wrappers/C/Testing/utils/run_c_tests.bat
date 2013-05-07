
echo "Running tests..."
@echo off 
REM ===  Call with three arguments!!
set compiler=%1
set compiler_version=%2
set build_type=%3


set install_folder=r:\installs\%compiler%\%compiler_version%\%build_type%
set model_folder=r:\models
set wc=r:\rrl
if %build_type% == release (
set report_file=%wc%\reports\%compiler%\%compiler_version%\c_tests.xml
) else (
set report_file=%wc%\reports\%compiler%\%compiler_version%\c_tests_%build_type%.xml
)


set temp_folder=r:\temp\%compiler_version%

c_api_tests.exe -m%model_folder% -r%report_file% -t%temp_folder% 

echo done...
