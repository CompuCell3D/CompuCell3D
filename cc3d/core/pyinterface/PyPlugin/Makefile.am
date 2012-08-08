


INC = -I$(COMPUCELL_SRC_DIR)/src/include \
      -I$(COMPUCELL_SRC_DIR)/src \
      -I$(COMPUCELL_SRC_DIR) \
      -I$(COMPUCELL_SRC_DIR)/src/CompuCell3D  \
      -I$(COMPUCELL_SRC_DIR)/src/CompuCell3D/plugins  \
      -I$(XERCESCROOT)/include \
      $(PYTHON_CPPFLAGS)

#LIB = -L$(COMPUCELL_INSTALL_DIR)/lib \
#      -L$(COMPUCELL_SRC_DIR)/lib\
#      -L$(XERCESCROOT)/lib


#LIBS = -lCompuCell3D  \
#       -lXercesUtils \
#       -lBasicUtils \
#       -lXMLCereal \
#       -lpthread \
#       -lxerces-c \
#       -lz \
#       -ldl




INCLUDES = -I$(top_srcdir)/. $(INC) \
	$(all_includes)
METASOURCES = AUTO
libPyPlugin_la_LDFLAGS = -module
noinst_LTLIBRARIES = libPyPlugin.la
libPyPlugin_la_SOURCES = PyPlugin.cpp PyCompuCellObjAdapter.cpp EnergyFunctionPyWrapper.cpp ChangeWatcherPyWrapper.cpp StepperPyWrapper.cpp TypeChangeWatcherPyWrapper.cpp  NeighborFinderParams.cpp PyAttributeAdder.cpp

noinst_HEADERS =  PyPlugin.h PyCompuCellObjAdapter.h EnergyFunctionPyWrapper.h ChangeWatcherPyWrapper.h StepperPyWrapper.cpp TypeChangeWatcherPyWrapper.h  NeighborFinderParams.h PyAttributeAdder.h
