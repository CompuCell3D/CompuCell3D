INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/CompuCell3D/plugins/Viscosity \
	-I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libViscosity_la_LDFLAGS = -module
plugin_LTLIBRARIES = libViscosity.la
viscosityincludedir=$(pluginincludedir)/Viscosity
viscosityinclude_HEADERS = ViscosityPlugin.h ViscosityEnergy.h
libViscosity_la_SOURCES = ViscosityPlugin.cpp ViscosityEnergy.cpp \
	ViscosityPluginProxy.cpp
libViscosity_la_LIBADD =\
	$(top_builddir)/src/PublicUtilities/libPublicUtilities.la
