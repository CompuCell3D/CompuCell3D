cmake_minimum_required(VERSION 2.8)
PROJECT(tests)

set(tests
Test_1.dat
Test_1.xml
Test_1.txt
ModelSourceTest.h
ModelSourceTest.c
)


install(
FILES ${tests}
DESTINATION testing 	COMPONENT testing
)

