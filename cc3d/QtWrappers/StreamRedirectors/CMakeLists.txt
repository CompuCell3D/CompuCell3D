SET(SRCS CustomStreamBuffers.cpp)

SET(MOC_HEADERS 

    QSender.h

)

#include all headers that need to be processed by moc
QT4_WRAP_CPP(MOC_FILES ${MOC_HEADERS})    

ADD_SHARED_LIBRARY(StreamRedirectors ${SRCS} ${MOC_FILES} LINK_LIBRARIES ${QT_LIBS} )



