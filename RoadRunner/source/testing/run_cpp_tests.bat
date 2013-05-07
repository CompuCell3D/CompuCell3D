
echo "Running tests..."
@echo off 

REM ===  Call with three arguments, eg vs 2010 Release..
set compiler=%1
set compiler_version=%2
set build_type=%3

set install_folder=r:\installs\%compiler%\%compiler_version%\%build_type%
set model_folder=r:\models
set wc=r:\rrl
set report_file=%wc%\reports\%compiler%\%compiler_version%\cpp_test_%build_type%.xml
set temp_folder=r:\rrTemp\%compiler_version%

cxx_api_tests.exe -m%model_folder% -r%report_file% -t%temp_folder% 

echo done...
