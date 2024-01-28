set(surface_dep_libs cc3d::SurfaceTracker)

if(APPLE)
    set(surface_dep_libs )
endif()

ADD_COMPUCELL3D_PLUGIN(Surface LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} ${surface_dep_libs})

