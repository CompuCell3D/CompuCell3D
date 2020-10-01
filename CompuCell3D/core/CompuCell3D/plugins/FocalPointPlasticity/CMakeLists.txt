ADD_COMPUCELL3D_PLUGIN(FocalPointPlasticity LINK_LIBRARIES ${PLUGIN_DEPENDENCIES} EXTRA_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})

SET (HEADER_FILES
	FocalPointPlasticityLinkInventoryBase.h
	FocalPointPlasticityLinkInventory.h
	FocalPointPlasticityLinks.h
)
