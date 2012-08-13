#include <Display3D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkActor2DCollection.h>
#include <vtkRenderer.h>
#include "vtkCylinderSource.h"
#include <vtkPolyDataMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkUnsignedShortArray.h>
#include <vtkPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkPointData.h>
#include <vtkStructuredPoints.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkVolumeTextureMapper2D.h>
#include <vtkVolumeTextureMapper3D.h>
#include <vtkPiecewiseFunction.h>
#include <vtkMarchingCubes.h>
#include <vtkContourFilter.h>
#include <vtkDiscreteMarchingCubes.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkFloatArray.h>
#include <vtkLookupTable.h>
#include <vtkScalarBarActor.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkRenderLargeImage.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGlyph3D.h>
#include <vtkConeSource.h>
#include <vtkOutlineFilter.h>



Display3D::~Display3D(){
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3D::initializeDisplay3D(){
      concMapper=0;
   concActor=0;

   vectorActor=0;
   vectorMapper=0;
   vectorGlyph=0;
   coneSource=0;

   outlineFilter=0;
   outlineMapper=0;
   outlineActor=0;
   
     
     


//    initialized=false;

//     qvtkWidget=new QVTKWidget(parent);
    qvtkWidget=dynamic_cast<QVTKWidget*>(this);
   

     
     renwin = vtkRenderWindow::New();
     renwin->StereoCapableWindowOn();
//      renwin->OffScreenRenderingOn();
     qvtkWidget->SetRenderWindow(renwin); //uncomment
     wimFilter=vtkWindowToImageFilter::New();
     pngWriter=vtkPNGWriter::New();


     qvtkWidget->setVisible(false);
//      renwin->Delete();

  // add a renderer
   ren = vtkRenderer::New();
/*   activeCamera=ren->GetActiveCamera();
   ren->ResetCamera();*/
//    camera=vtkCamera::New();
//    ren->SetActiveCamera(camera);

   qvtkWidget->GetRenderWindow()->AddRenderer(ren);//uncomment

   qvtkWidget->resize( QSize(500, 500).expandedTo(qvtkWidget->minimumSizeHint()) );
    ren->SetBackground(1,1,1);

}

Display3D::Display3D(QWidget *parent , const char *name ):QVTKWidget(parent),Display3DBase(name){
   cerr<<"INSIDE Display3D constructor "<<endl;
   initializeDisplay3D();
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////