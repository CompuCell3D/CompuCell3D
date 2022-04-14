SET(LIBS cc3d::muParser cc3d::XMLUtils)
SET(SRCS ExpressionEvaluator.cpp )

ADD_SHARED_LIBRARY(ExpressionEvaluator ${SRCS} LINK_LIBRARIES ${LIBS})
TARGET_INCLUDE_DIRECTORIES(${ExpressionEvaluatorLib} PUBLIC ${INCLUDE_DIRS})

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/muParser/ExpressionEvaluator .h
   ExpressionEvaluator
   ExpressionEvaluatorDLLSpecifier   
)
