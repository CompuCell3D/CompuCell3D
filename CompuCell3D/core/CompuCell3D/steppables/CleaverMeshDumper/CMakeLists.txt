
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)

ADD_COMPUCELL3D_STEPPABLE(CleaverMeshDumper
  CleaverMeshDumper.cpp
  CleaverMeshDumperProxy.cpp LINK_LIBRARIES ${STEPPABLE_DEPENDENCIES} cleaver dolfin)


ADD_COMPUCELL3D_STEPPABLE_HEADERS(CleaverMeshDumper
  CleaverMeshDumper.h
  )        
