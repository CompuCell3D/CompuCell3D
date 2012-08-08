INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/CompuCell3D/plugins/Stretchness \
	-I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libStretchness_la_LDFLAGS = -module
plugin_LTLIBRARIES = libStretchness.la
stretchnessincludedir=$(pluginincludedir)/Stretchness
stretchnessinclude_HEADERS = StretchnessEnergy.h StretchnessPlugin.h
libStretchness_la_SOURCES = StretchnessEnergy.cpp StretchnessPlugin.cpp\
	StretchnessPluginProxy.cpp
