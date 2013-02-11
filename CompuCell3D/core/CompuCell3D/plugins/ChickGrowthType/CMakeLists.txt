

#had to add H infront of ChickGrowthType to ensure that this library will be loaded AFTER CC3DGrowth ono which it depends. 
#will need to come up with better solution for the way in which plugins are handled

# better solution is to have  PATH or LIBPATH to poin into location where all the plugins are so that loader can find dependencies
ADD_COMPUCELL3D_PLUGIN(ChickGrowthType LINK_LIBRARIES CompuCellLibShared GrowthShared)


