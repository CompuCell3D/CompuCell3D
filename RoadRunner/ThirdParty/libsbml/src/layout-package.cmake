###############################################################################
#
# $URL: https://sbml.svn.sourceforge.net/svnroot/sbml/trunk/libsbml/src/layout-package.cmake $
# $Id: layout-package.cmake 16010 2012-07-14 00:27:36Z mhucka $
#
# Description       : CMake include file for SBML Level 3 Layout package
# Original author(s): Frank Bergmann <fbergman@caltech.edu>
# Organization      : California Institute of Technology
#
# This file is part of libSBML.  Please visit http://sbml.org for more
# information about SBML, and the latest version of libSBML.
#
# Copyright (C) 2009-2012 jointly by the following organizations: 
#     1. California Institute of Technology, Pasadena, CA, USA
#     2. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
#  
# Copyright (C) 2006-2008 by the California Institute of Technology,
#     Pasadena, CA, USA 
#  
# Copyright (C) 2002-2005 jointly by the following organizations: 
#     1. California Institute of Technology, Pasadena, CA, USA
#     2. Japan Science and Technology Agency, Japan
# 
# This library is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation.  A copy of the license agreement is provided
# in the file named "LICENSE.txt" included with this software distribution
# and also available online as http://sbml.org/software/libsbml/license.html
#
###############################################################################

if(ENABLE_LAYOUT)

add_definitions( -DUSE_LAYOUT )

#build up sources
set(LAYOUT_SOURCES)
# go through all directtories: common, extension, sbml and util
foreach(dir common extension sbml util)

	
	# file sources
	file(GLOB current ${CMAKE_CURRENT_SOURCE_DIR}/sbml/packages/layout/${dir}/*.cpp
	                  ${CMAKE_CURRENT_SOURCE_DIR}/sbml/packages/layout/${dir}/*.c
	                  ${CMAKE_CURRENT_SOURCE_DIR}/sbml/packages/layout/${dir}/*.h)
	
	# add sources 
	set(LAYOUT_SOURCES ${LAYOUT_SOURCES} ${current})

	# mark header files for installation 
	file(GLOB layout_headers ${CMAKE_CURRENT_SOURCE_DIR}/sbml/packages/layout/${dir}/*.h)
    install(FILES ${layout_headers} DESTINATION include/sbml/packages/layout/${dir})	
	
endforeach()

# create source group for IDEs
source_group(layout_package FILES ${LAYOUT_SOURCES})

# add layout sources to SBML sources
SET(LIBSBML_SOURCES ${LIBSBML_SOURCES} ${LAYOUT_SOURCES})

####################################################################
#
# add test scripts
#
if(WITH_CHECK)

	add_subdirectory(sbml/packages/layout/sbml/test)

endif()

endif()
