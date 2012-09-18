INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core/muParser
    )

if(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
    SET(LIBS BasicUtilsStatic muParserStatic XMLUtilsShared)
else(${CMAKE_SYSTEM_NAME} STREQUAL Windows)
   SET(LIBS BasicUtilsShared muParserShared XMLUtilsShared)
endif(${CMAKE_SYSTEM_NAME} STREQUAL Windows)

SET(SRCS ExpressionEvaluator.cpp )

ADD_SHARED_LIBRARY(ExpressionEvaluator ${SRCS} LINK_LIBRARIES ${LIBS})

INSTALL_FILES(/include/CompuCell3D/muParser/ExpressionEvaluator .h
   ExpressionEvaluator
   ExpressionEvaluatorDLLSpecifier   
)


