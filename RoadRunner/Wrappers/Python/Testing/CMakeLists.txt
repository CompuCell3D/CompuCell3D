CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(rrWrappers)

#Some of the tests sets a temp folder thats on the same level as the 'bin' folder. Create it
FILE(MAKE_DIRECTORY temp) 

install (
DIRECTORY "temp" DESTINATION "." 
DIRECTORY_PERMISSIONS 
    OWNER_WRITE OWNER_READ OWNER_EXECUTE
    GROUP_WRITE GROUP_READ GROUP_EXECUTE
    WORLD_WRITE WORLD_READ WORLD_EXECUTE
)
 

#=============== INSTALL =====================
install(
FILES 
rrTestCompiler.py
rrRunTests.py
rrTester.py
rrDebug.py 
DESTINATION python  	COMPONENT testing
)

