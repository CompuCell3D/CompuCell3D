SET(SRCS CC3DXMLElement.cpp CC3DXMLElementWalker.cpp)

find_library(EXPAT_LIBRARY_LOCAL NAMES expat libexpat HINTS ${EXPAT_LIBRARIES})

message("expat library local ${EXPAT_LIBRARY_LOCAL}")

ADD_SHARED_LIBRARY(XMLUtils ${SRCS} LINK_LIBRARIES cc3d::PublicUtilities)
TARGET_INCLUDE_DIRECTORIES(${XMLUtilsLib} PUBLIC
    $<BUILD_INTERFACE:${COMPUCELL3D_SOURCE_DIR}/core>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${COMPUCELL3D_INSTALL_INCLUDE_DIR}>
)

INSTALL_FILES(/${COMPUCELL3D_INSTALL_INCLUDE_DIR}/XMLUtils .h
   CC3DXMLElement
   CC3DXMLElementWalker
   XMLParserExpatLibDLLSpecifier
   XMLUtilsDLLSpecifier   
   CC3D_XMLUtils
)

if(BUILD_CPP_ONLY_EXECUTABLE)

    SET(SRCS_XML_PARSER_EXPAT_LIB
        XMLParserExpat.cpp
    )

    ADD_SHARED_LIBRARY(XMLParserExpatLib ${SRCS_XML_PARSER_EXPAT_LIB} LINK_LIBRARIES ${EXPAT_LIBRARY_LOCAL} cc3d::XMLUtils)

endif(BUILD_CPP_ONLY_EXECUTABLE)
