INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libCellVelocity_la_LDFLAGS = -module
plugin_LTLIBRARIES = libCellVelocity.la
cellvelocityincludedir=$(pluginincludedir)/CellVelocity
plugininclude_HEADERS = CellVelocityPlugin.h CellVelocityData.h \
		CellVelocityDataAccessor.h CellVelocityDataClassFactory.h CellInstantVelocityPlugin.h
libCellVelocity_la_SOURCES = CellVelocityPlugin.cpp CellVelocityData.cpp \
		CellVelocityPluginProxy.cpp CellInstantVelocityPlugin.cpp CellInstantVelocityPluginProxy.cpp
libCellVelocity_la_LIBADD =\
	$(top_builddir)/src/PublicUtilities/libPublicUtilities.la
