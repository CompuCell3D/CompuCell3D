INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/CompuCell3D/plugins/ContactLocalFlex \
	-I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libContactLocalFlex_la_LDFLAGS = -module
plugin_LTLIBRARIES = libContactLocalFlex.la
contactlocalflexincludedir=$(pluginincludedir)/ContactLocalFlex
contactlocalflexinclude_HEADERS = ContactLocalFlexEnergy.h ContactLocalFlexPlugin.h ContactLocalFlexData.h
libContactLocalFlex_la_SOURCES = ContactLocalFlexEnergy.cpp ContactLocalFlexPlugin.cpp ContactLocalFlexPluginProxy.cpp
