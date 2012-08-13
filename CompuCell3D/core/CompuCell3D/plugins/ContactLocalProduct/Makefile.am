INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/CompuCell3D/plugins/ContactLocalProduct \
	-I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO
libContactLocalProduct_la_LDFLAGS = -module
plugin_LTLIBRARIES = libContactLocalProduct.la
contactlocalproductincludedir=$(pluginincludedir)/ContactLocalProduct
contactlocalproductinclude_HEADERS = ContactLocalProductEnergy.h ContactLocalProductPlugin.h ContactLocalProductData.h
libContactLocalProduct_la_SOURCES = ContactLocalProductEnergy.cpp ContactLocalProductPlugin.cpp ContactLocalProductPluginProxy.cpp
