CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3 FATAL_ERROR)
PROJECT(RR_PYTHON_APPS)


if(${MSVC})
endif(${MSVC})

install(    
            FILES 
            rr.py
            DESTINATION python 	
            COMPONENT Python
       )

