INCLUDES = -I$(top_srcdir)/. -I$(top_srcdir)/src \
	-I$(top_srcdir)/src/CompuCell3D -I$(top_srcdir)/src/include $(all_includes)
METASOURCES = AUTO

plugin_LTLIBRARIES = libCellBoundaryTracker.la
libCellBoundaryTracker_la_SOURCES = CellBoundaryTrackerPlugin.cpp \
		CellBoundaryTrackerPluginProxy.cpp CellBoundaryTracker.cpp CellBoundaryAlgorithmDescription.txt

cellboundarytrackerincludedir=$(pluginincludedir)/CellBoundaryTracker
cellboundarytrackerinclude_HEADERS = CellBoundaryTrackerPlugin.h CellBoundaryTracker.h
