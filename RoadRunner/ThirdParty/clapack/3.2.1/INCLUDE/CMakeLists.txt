CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(clapack_headers)

set(headers 
f2c.h
clapack.h
blaswrap.h
)

install(FILES ${headers} 
        DESTINATION include/clapack)


