


# the variable "SRCS" contains all .cpp files of this project
set(SRCS
    imagewindowBase.cpp 
    imagewindow.cpp 
    imagewindow_NOX.cpp 
    main.cpp 
    transactionthread.cpp 
    ColorItem.cpp 
    Graphics2DBase.cpp 
    Graphics2D.cpp 
    Graphics2D_NOX.cpp 
    contourlines.cpp 
    Display3D.cpp 
    Display3DBase.cpp     
    Display3D_NOX.cpp     
    GraphicsBase.cpp 
    ScreenshotData.cpp 
    FileUtils.cpp 
    UniversalGraphicsSettings.cpp														
)
	
qt4_automoc(${SRCS})
 
SET(LIBS
         simthreadShared
         CompuCellLibShared
         CC3DPublicUtilities
         CC3DBasicUtils 
         # CC3DXMLCereal 
         # CC3DXercesUtils 
         CC3DField3D 
         CC3DPotts3D 
         CC3DBoundary
         CC3DAutomaton
         ${PYTHON_LIBRARIES} 
         # ${XERCESC_LIBRARY}
   )
 

# tell cmake to create .moc files for all files in the variable qt_app_SRCS that require such a file.
# note: this assumes that you use #include "header.moc" in your files





#this is how we include image collection in cmake and qt4
QT4_ADD_RESOURCES(player_rc_srcs player.qrc)

SET(FORMS_UI
    CellTypeColorConfigure.ui 
    ColormapPlotConfigure.ui 
    ScreenshotFrequencyConfigure.ui 
    CellTypeColorConfigure.ui 
    VectorFieldPlotConfigure.ui 
    Configure3DDialog.ui 
    PythonConfigureDialog.ui
    TypesThreeDConfigure.ui
    SimulationFileOpenDialog.ui
)

SET(FORMS_UI_CPP
    ColormapPlotConfigure.cpp 
    ScreenshotFrequencyConfigure.cpp
    CellTypeColorConfigure.cpp 
    VectorFieldPlotConfigure.cpp 
    Configure3DDialog.cpp 
    PythonConfigureDialog.cpp
    TypesThreeDConfigure.cpp
    SimulationFileOpenDialog.cpp
)

#generate necessary files from ui files
QT4_WRAP_UI(FORMS_UI_SOURCES  ${FORMS_UI} )


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PYTHON_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/core/PublicUtilities ${CMAKE_SOURCE_DIR}/core/CompuCellPlayer/QVTKWidget ${VTK_INCLUDE_DIRS})



#notice , here we include header files that accompany FORMS_UI_CPP files
SET(MOC_HEADERS 
   imagewindow.h
   Display3D.h
   ColormapPlotConfigure.h
   ScreenshotFrequencyConfigure.h
   CellTypeColorConfigure.h
   VectorFieldPlotConfigure.h
   Configure3DDialog.h
   PythonConfigureDialog.h
   TypesThreeDConfigure.h
   SimulationFileOpenDialog.h

)

#include all headers that need to be processed by moc
QT4_WRAP_CPP(MOC_FILES ${MOC_HEADERS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})




#message("this is a qt_open_gl ${QT_QTOPENGL_LIBRARY}")
#message("this is a open_gl ${OPENGL_LIBRARIES}")


# create an executable file named "qt_app" from the source files in the variable "qt_app_SRCS".
#add_executable(CompuCellPlayer ${SRCS} ${FORMS_UI_SOURCES} ${MOC_FILES})

add_executable(CompuCellPlayer  ${SRCS} ${FORMS_UI_CPP} ${player_rc_srcs} ${FORMS_UI_SOURCES} ${MOC_FILES})
# add_executable(CompuCellPlayer ${SRCS} ${MOC_FILES})
# add_executable(CompuCellPlayer image_collection.cpp ${SRCS} ${FORMS_UI_SOURCES} ${MOC_FILES})

# add_executable(CompuCellPlayer image_collection.cpp ${SRCS} ${FORMS_UI_H} ${MOC_FILES})
	
# link the "qt_app" target (i.e. the executable file added above) agains the Qt
target_link_libraries(CompuCellPlayer ${QT_LIBRARIES} QVTK  ${QT_QTOPENGL_LIBRARY} ${OPENGL_LIBRARIES} ${LIBS} vtkVolumeRendering vtkHybrid vtkFiltering vtkCommon)

install_targets(/bin CompuCellPlayer)	