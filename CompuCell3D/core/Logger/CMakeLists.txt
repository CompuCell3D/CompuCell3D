# Recurse into subprojects

SET(SRCS
  Logger.cpp
  )

INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}
    ${COMPUCELL3D_SOURCE_DIR}/core/Logger
    )

ADD_STATIC_LIBRARY(LoggerLib ${SRCS}  )

# INSTALL_FILES(/include/muParser .h
INSTALL_FILES(/include/CompuCell3D/Logger .h
Logger

)




