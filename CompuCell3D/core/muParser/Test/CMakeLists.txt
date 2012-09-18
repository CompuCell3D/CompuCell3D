  INCLUDE_DIRECTORIES ( 
    ${COMPUCELL3D_SOURCE_DIR}/core/muParser
    )
  
  ADD_EXECUTABLE(muParserTest main.cpp)
  TARGET_LINK_LIBRARIES(muParserTest
   muParserStatic
   XMLUtilsShared

  )
 INSTALL_TARGETS(/bin muParserTest)
