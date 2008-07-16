@echo off
set PATH=C:\Program Files\classes\CalculatorTry;%PATH%
set PATH=c:\Program Files\Java\jdk1.6.0_06\jre\bin\server\;%PATH%
rem set CLASSPATH=C:\Program Files\classes\Geometry;%CLASSPATH%

rem this is necessary to point c++ to java classes
set CLASSPATH=C:\JavaProjects\Shapes\bin\;C:\Program Files\classes\CalculatorTry;%CLASSPATH%


rem set jython_module_path="C:\JavaProjects\Shapes\bin"
rem set jython_module_path="C:\Program Files\classes\CalculatorTry"

javac CalculatorTry/*.java
rem javac Geometry/*.java

rem jython.bat -Dpython.path=%jython_module_path% test_j.py
jython.bat  test_j.py
