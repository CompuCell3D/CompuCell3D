INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libConcentration_la_LDFLAGS = -module
lib_LTLIBRARIES = libConcentration.la

concentrationincludedir=$(pluginincludedir)/Concentration
concentrationinclude_HEADERS = ConcentrationPlugin.h Concentration.h
libConcentration_la_SOURCES = ConcentrationPlugin.cpp ConcentrationPluginProxy.cpp
