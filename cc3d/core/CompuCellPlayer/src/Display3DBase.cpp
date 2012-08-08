// #include <qcolordialog.h>

#include "Display3DBase.h"
#include <map>
#include <set>
//#include <QtGui>
//#include <QtOpenGL>


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

#include "Projection2DData.h"

#include <GraphicsDataFields.h>
#include "UniversalGraphicsSettings.h"



#include <iostream>
#include <limits>



using namespace std;






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::QColorToVTKColor(const QColor & color, double * _vtkColorArray){
      
      
      QColor white("white");
      _vtkColorArray[0]=color.red()/(float)white.red();
      _vtkColorArray[1]=color.green()/(float)white.green();
      _vtkColorArray[2]=color.blue()/(float)white.blue();

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::repaintVTK(){

//    cerr<<"Before Render"<<endl;   
//    sleep(2);
   renwin->Render();
//    cerr<<"After Render"<<endl;
//    sleep(2);
   

}

void Display3DBase::setCameraRotations(const Configure3DData & _data3DConf){
   

   if(_data3DConf.rotationX != mostRecent3DConf.rotationX 
      || _data3DConf.rotationY != mostRecent3DConf.rotationY 
      || _data3DConf.rotationZ != mostRecent3DConf.rotationZ 
   ){
         mostRecent3DConf=_data3DConf;
/*         mostRecent3DConf.rotationX=_data3DConf.rotationX;
         mostRecent3DConf.rotationY=_data3DConf.rotationY;
         mostRecent3DConf.rotationZ=_data3DConf.rotationZ;*/
    
   
      //this will reset camera position to default
      activeCamera=ren->GetActiveCamera();
//       double dirProj[3];
//       activeCamera->GetDirectionOfProjection(dirProj);
//       activeCamera->Print(cerr);
//       cerr<<"dirProj[0]="<<acos(dirProj[0])<<" dirProj[1]="<<acos(dirProj[1])<<"dirProj[0]="<<acos(dirProj[2])<<endl;
      
   
      activeCamera->SetViewUp(0, 0, 1);
      activeCamera->SetPosition(0, 1, 0);
      activeCamera->SetFocalPoint(0, 0, 0);
   
   
      //this will rotate camera. Notice that if you do not reset camera position as above the rotation will be relative to last position. 
      //For some reason when all the rotations are zero screenshots are not output correctly. Will fix it later

      activeCamera->Elevation(_data3DConf.rotationX);
      activeCamera->Azimuth(_data3DConf.rotationY);
      activeCamera->Roll(_data3DConf.rotationZ);

      ren->ResetCamera();
      repaintVTK();
   }else{

      return;
   }


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::initializeDisplay3D(){

}
Display3DBase::~Display3DBase(){}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Display3DBase::Display3DBase(const char *name)
    : GraphicsBase()
{

   initializeDisplay3D();

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::produceImage(const std::string & _fileName){      

      vtkRenderLargeImage *renderLarge;
      renderLarge=vtkRenderLargeImage::New();
      pngWriter=vtkPNGWriter::New();
      renderLarge->SetInput(ren);
      renderLarge->SetMagnification(1);
      pngWriter->SetInputConnection(renderLarge->GetOutputPort());
      pngWriter->SetFileName(_fileName.c_str());
      cerr<<" \n\n\n writing "<<_fileName<<"\n\n\n"<<endl;
      pngWriter->Write();
      renderLarge->Delete();
      pngWriter->Delete();

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::setSizeLMN(unsigned int _sizeL , unsigned int _sizeM ,unsigned int _sizeN){
   sizeL=_sizeL;
   sizeM=_sizeM;
   sizeN=_sizeN;

   maxDimension=sizeL;
   
   if(sizeM>maxDimension)
      maxDimension=sizeM;
   if(sizeN>maxDimension)
      maxDimension=sizeN;

   data3DConf.sizeX= sizeL;
   data3DConf.sizeY= sizeM;
   data3DConf.sizeZ= sizeN;

            
   
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::setInitialConfigure3DData(const Configure3DData & _data){

   sizeL=_data.sizeL;
   sizeM=_data.sizeM;
   sizeN=_data.sizeN;

   maxDimension=sizeL;
   if(sizeM>maxDimension)
      maxDimension=sizeM;
   if(sizeN>maxDimension)
      maxDimension=sizeN;

   data3DConf.sizeX= _data.sizeX;
   data3DConf.sizeY= _data.sizeY;
   data3DConf.sizeZ= _data.sizeZ;

   data3DConf.rotationX= _data.rotationX;
   data3DConf.rotationY= _data.rotationY;
   data3DConf.rotationZ= _data.rotationZ;

   //building oncentration LookupTable - blue lowest, red highest values

   colorLookupTable=vtkLookupTable::New();
   colorLookupTable->SetHueRange(0.6666669999999999,0);
   colorLookupTable->SetSaturationRange(1.,1.);
   colorLookupTable->SetValueRange(1.,1.);
   colorLookupTable->SetAlphaRange(1.,1.);
   colorLookupTable->SetNumberOfColors(1024);
   colorLookupTable->Build();

   //creating legendActor
   legendActor=vtkScalarBarActor::New();

  scalarsUShort = vtkUnsignedShortArray::New();
  points = vtkPoints::New();
//   points->Allocate(data3DConf.sizeX*data3DConf.sizeY*data3DConf.sizeY);
  vectorGrid=vtkUnstructuredGrid::New();
  vectorGrid->SetPoints(points);

  vectors=vtkFloatArray::New();
  vectors->SetNumberOfComponents(3);
  vectors->SetName("vectors");

  vectorGrid->SetPoints(points);
  vectorGrid->GetPointData()->SetVectors(vectors);


      cerr<<"VTK DISPLAY3D INITIALIZATION"<<endl;

      dims[0]=data3DConf.sizeX;
      dims[1]=data3DConf.sizeY;
      dims[2]=data3DConf.sizeZ;

//       points->Delete();


      vol=vtkStructuredPoints::New();
//       vol->SetDimensions(dims[0],dims[1],dims[2]);
      initializeDataSetSize(vol,dims[0],dims[1],dims[2] );
      if(latticeType==CompuCell3D::HEXAGONAL_LATTICE)
         vol->SetSpacing(1.0,sqrt(3.0)/2.0,sqrt(6.0)/3.0);
//       vol->SetDimensions(dims[0],dims[1],dims[2]);
//        vol->GetPointData()->SetScalars(scalars);
      vol->GetPointData()->SetScalars(scalarsUShort);

      concVol=vtkStructuredPoints::New();
      if(latticeType==CompuCell3D::HEXAGONAL_LATTICE)
         concVol->SetSpacing(1.0,sqrt(3.0)/2.0,sqrt(6.0)/3.0);
      initializeDataSetSize(concVol,dims[0],dims[1],dims[2] );
//       concVol->SetDimensions(dims[0],dims[1],dims[2]);


      concArray=vtkFloatArray::New();
      concArray->SetName("concentration");
//       typeContourArray=vtkFloatArray::New();
//       typeContourArray->SetName("ctypeArray");

      concVol->GetPointData()->SetScalars(scalarsUShort);
      concVol->GetPointData()->AddArray(dynamic_cast<vtkDataArray*>(concArray));

      cerr<<"scalarsUShort->GetDataType() "<<scalarsUShort->GetDataType()<<endl;
      cerr<<"scalarsUShort->GetNumberOfComponents() "<<scalarsUShort->GetNumberOfComponents()<<endl;
      

   
      outlineFilter=vtkOutlineFilter::New();
      outlineFilter->SetInput(vol);
      outlineMapper=vtkPolyDataMapper::New();
      outlineMapper->SetInputConnection(outlineFilter->GetOutputPort());
      outlineActor=vtkActor::New();
      outlineActor->SetMapper(outlineMapper);
       
      



}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::initializeDataSetSize(vtkStructuredPoints * dataSet, int _xDim,int _yDim,int _zDim ){

   int xDimLocal=_xDim;
   int yDimLocal=_yDim;
   int zDimLocal=_zDim;

   if(_xDim==1){
      xDimLocal+=2;
   }

   if(_yDim==1){
      yDimLocal+=2;
   }

   if(_zDim==1){
      zDimLocal+=2;
   }

   dataSet->SetDimensions(xDimLocal,yDimLocal,zDimLocal);
   


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::initializeGL()
{

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::setDimensions(GLfloat _sizeX,GLfloat _sizeY, GLfloat _sizeZ ){


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::resizeGL(int width, int height)
{

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::paintGL()
{

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::initializeVTKSettings(){

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::initDraw(){

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::draw()
{


}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Display3DBase::faceAtPosition(const QPoint &pos)
{
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::drawCube(GLfloat x, GLfloat y , GLfloat z , GLfloat side){

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::drawCube(GLfloat x, GLfloat y , GLfloat z , GLfloat side, const QColor &color){

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::drawSquare(GLfloat x, GLfloat y , GLfloat z , GLfloat side, const QColor &color){

      
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::drawRectangle(GLfloat x, GLfloat y , GLfloat z , GLfloat width , GLfloat height, const QColor &color){


}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::removeCurrentActors(){

   vtkActorCollection * actorCollection=ren->GetActors();
   int numberOfCurrentActors=actorCollection->GetNumberOfItems();

   for(int aCount = 0 ; aCount < numberOfCurrentActors ; ++aCount){
      ren->RemoveActor(actorCollection->GetLastActor());
   }

   
   vtkActor2DCollection * actor2DCollection=ren->GetActors2D();
   int numberOfCurrentActors2D=actor2DCollection->GetNumberOfItems();
   
   for(int aCount = 0 ; aCount < numberOfCurrentActors2D ; ++aCount){
      ren->RemoveActor(actor2DCollection->GetLastActor2D());
   }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::fillEmptyCellTypeData(vtkUnsignedShortArray * _scalarsUShort, unsigned int & _offset){

      	
   for(int z=0 ; z< data3DConf.sizeZ ; ++z)
      for(int y=0 ; y< data3DConf.sizeY ; ++y)
         for(int x=0 ; x< data3DConf.sizeX ; ++x)
         {
         
            _scalarsUShort->InsertValue(_offset,0);
            ++_offset;
         }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::fillEmptyConcentrationData(vtkFloatArray *_concArray, unsigned int & _offset){


      for(int z=0 ; z< data3DConf.sizeZ ; ++z)
         for(int y=0 ; y< data3DConf.sizeY ; ++y)
            for(int x=0 ; x< data3DConf.sizeX ; ++x){
         
            _concArray->InsertValue(_offset,0.0);

            ++_offset;
         }


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Display3DBase::paintLattice(){


   if(!drawingAllowed)
   return;

   

   field3DGraphicsDataPtr = & graphFieldsPtr->field3DGraphicsData;
   drawFcnPtr=&Display3DBase::paint3DLattice;

   field3DGraphicsData_t & field3DGraphicsData = * field3DGraphicsDataPtr;
   

   unsigned short maxType=0;
   unsigned short cellType;
   set<unsigned short> distinctCellTypes;

   unsigned int offset=0;
   if(data3DConf.sizeX==1||data3DConf.sizeY==1||data3DConf.sizeZ==1){
      fillEmptyCellTypeData(scalarsUShort, offset);
   }

   
     for(int z=0 ; z< data3DConf.sizeZ ; ++z)
         for(int y=0 ; y< data3DConf.sizeY ; ++y)
            for(int x=0 ; x< data3DConf.sizeX ; ++x)
	   {
         


            cellType=field3DGraphicsData[x][y][z].type;
            distinctCellTypes.insert(cellType);

            if(cellType>maxType){
               maxType=cellType;
            }
            if ( ! univGraphSetPtr->avoidType(cellType) ){//only if cell types are not invisible
               scalarsUShort->InsertValue(offset,cellType);
            }else{
               scalarsUShort->InsertValue(offset,0);//treat invisible cell types as medium
            }


            ++offset;
         }

   if(data3DConf.sizeX==1||data3DConf.sizeY==1||data3DConf.sizeZ==1){
      fillEmptyCellTypeData(scalarsUShort, offset);
   }


         removeCurrentActors();

//          cerr<<"maxType="<<maxType<<endl;
//          cerr<<"size of typeActorsVec before"<<typeActorsVec.size()<<endl;
         //Now will check how many actors there are in the vector of cell TypeActors;
         if(typeActorsVec.size()==0 || typeActorsVec.size()-1<maxType){
            //will have to insert more actors, filters, mappers etc into vectors

            int offset=(maxType-typeActorsVec.size())+1;
            for(int i = 0 ; i < offset ; ++i){
//                cerr<<"i = "<<i<<endl;
               typeActorsVec.push_back(0);
               typeExtractorVec.push_back(0);
               smootherFilterVec.push_back(0);
               polyDataNormalsVec.push_back(0);
               typeExtractorMapperVec.push_back(0);

            }
         }

//       cerr<<"size of typeExtractorVec "<<typeExtractorVec.size()<<endl;
      //now will inte3rate over distinct types and create actors,filters mappers for types that do have them yet
      for(set<unsigned short>::iterator sitr=distinctCellTypes.begin(); sitr!=distinctCellTypes.end() ;++sitr){
         cellType=*sitr;

         //filters got reallocated every time but they also get erased every time

         if ( ! univGraphSetPtr->avoidType(cellType) ){ //extract surface only if not invisible
//             cerr<<"cellType="<<cellType<<endl;
            typeExtractorVec[cellType]=vtkDiscreteMarchingCubes::New();
//             typeExtractorVec[cellType]=vtkContourFilter::New();
            
            
            if(!smootherFilterVec[cellType]){
               smootherFilterVec[cellType]=vtkSmoothPolyDataFilter::New();
            }

            if(!polyDataNormalsVec[cellType]){
               polyDataNormalsVec[cellType]=vtkPolyDataNormals::New();
            }

   
            if(!typeActorsVec[cellType]){
               typeActorsVec[cellType]=vtkActor::New();
            }
   
            if(!typeExtractorMapperVec[cellType]){
               typeExtractorMapperVec[cellType]=vtkPolyDataMapper::New();
            }

         }
      }

      UniversalGraphicsSettings::colorMapItr pos;
      QColor *colorPtr;
      double colorRGB[3];
      //set up vtk pipeline
     for(set<unsigned short>::iterator sitr=distinctCellTypes.begin(); sitr!=distinctCellTypes.end() ;++sitr){
         cellType=*sitr;

         if ( ! univGraphSetPtr->avoidType(cellType) ){//only if cell types are not invisible
            typeExtractorVec[cellType]->SetInput(vol);
            typeExtractorVec[cellType]->SetValue(0,cellType);
            smootherFilterVec[cellType]->SetInputConnection(typeExtractorVec[cellType]->GetOutputPort());
            polyDataNormalsVec[cellType]->SetInputConnection(smootherFilterVec[cellType]->GetOutputPort());
            polyDataNormalsVec[cellType]->SetFeatureAngle(45.0);
            typeExtractorMapperVec[cellType]->SetInputConnection(polyDataNormalsVec[cellType]->GetOutputPort());
            typeExtractorMapperVec[cellType]->ScalarVisibilityOff();
            typeActorsVec[cellType]->SetMapper(typeExtractorMapperVec[cellType]);
            

            
            //Now will color actors
            pos=univGraphSetPtr->typeColorMap.find(cellType);
            if(pos!=univGraphSetPtr->typeColorMap.end()){
               colorPtr=&(pos->second);
            }else{
               colorPtr=&univGraphSetPtr->defaultColor;
            }

            QColorToVTKColor(*colorPtr, colorRGB);
            typeActorsVec[cellType]->GetProperty()->SetDiffuseColor(colorRGB);

            ren->AddActor(typeActorsVec[cellType]);
        }
     }
      //setting backgroundColor to the color of Medium (type 0)
      pos=univGraphSetPtr->typeColorMap.find(0);
      if(pos!=univGraphSetPtr->typeColorMap.end()){
         colorPtr=&(pos->second);
      }else{
         colorPtr=&univGraphSetPtr->defaultColor;
      }

      QColorToVTKColor(*colorPtr, colorRGB);
      ren->SetBackground(colorRGB);



      for(set<unsigned short>::iterator sitr=distinctCellTypes.begin(); sitr!=distinctCellTypes.end() ;++sitr){
         cellType=*sitr;

         //filters got reallocated every time but they also get erased every time

         if ( ! univGraphSetPtr->avoidType(cellType) ){ //extract surface only if not invisible
//             cerr<<"cellType="<<cellType<<endl;
	    typeExtractorVec[cellType]->Delete();
            typeExtractorVec[cellType]=0;
//             typeExtractorVec[cellType]=vtkContourFilter::New();
            
            
            if(!smootherFilterVec[cellType]){
               	smootherFilterVec[cellType]->Delete();
	 	smootherFilterVec[cellType]=0;
            }

            if(!polyDataNormalsVec[cellType]){
               	polyDataNormalsVec[cellType]->Delete();
		polyDataNormalsVec[cellType]=0;
            }

   
            if(!typeActorsVec[cellType]){
               typeActorsVec[cellType]->Delete();
	       typeActorsVec[cellType]=0;
            }
   
            if(!typeExtractorMapperVec[cellType]){
               typeExtractorMapperVec[cellType]->Delete();
	       typeExtractorMapperVec[cellType]=0;
            }

         }
      }

      //here will delete filters

/*      for(set<unsigned short>::iterator sitr=distinctCellTypes.begin(); sitr!=distinctCellTypes.end() ;++sitr){
         cellType=*sitr;

         //filters got reallocated every time but they also get erased every time
         if ( ! univGraphSetPtr->avoidType(cellType) ){//only if cell types are not invisible
            typeExtractorVec[cellType]->Delete();


         }
      }*/
      
      repaintVTK();
      setCameraRotations(data3DConf); //this has callsResetCamera which in turns re-renders the scene. It has to be called after repaint


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::paintConcentrationLattice(){

   //Check if perhaps mappers and actors also have to be deleted after painting each scene 

   if(!drawingAllowed)
   return;


   floatField3D_t & concentrationField=*currentConcentrationFieldPtr;
   field3DGraphicsDataPtr = & graphFieldsPtr->field3DGraphicsData;
   drawFcnPtr=&Display3DBase::paint3DConcentrationLattice;

   field3DGraphicsData_t & field3DGraphicsData = * field3DGraphicsDataPtr;

   maxConcentrationTrue=(numeric_limits<float>::min)();
   minConcentrationTrue=(numeric_limits<float>::max)();

   unsigned short cellType;
   set<unsigned short> distinctCellTypes;
   unsigned int offset=0;
   unsigned int offset1=0;

   if(data3DConf.sizeX==1||data3DConf.sizeY==1||data3DConf.sizeZ==1){
      fillEmptyCellTypeData(scalarsUShort, offset);
      fillEmptyConcentrationData(concArray, offset1);
   }


   for(int z=0 ; z< data3DConf.sizeZ ; ++z)
      for(int y=0 ; y< data3DConf.sizeY ; ++y)
         for(int x=0 ; x< data3DConf.sizeX ; ++x){

            cellType=field3DGraphicsData[x][y][z].type;
            distinctCellTypes.insert(cellType);

            if ( ! univGraphSetPtr->avoidType(cellType) ){//only if cell types are not invisible
               scalarsUShort->InsertValue(offset,cellType);
            }else{
               scalarsUShort->InsertValue(offset,0);
            }


            if(concentrationField[x][y][z]>maxConcentrationTrue){
               maxConcentrationTrue=concentrationField[x][y][z];
            }

            if(concentrationField[x][y][z]<minConcentrationTrue){
               minConcentrationTrue=concentrationField[x][y][z];
            }

               concArray->InsertValue(offset,concentrationField[x][y][z]);

            
            ++offset;
            ++offset1;
         }

   if(data3DConf.sizeX==1||data3DConf.sizeY==1||data3DConf.sizeZ==1){
      fillEmptyCellTypeData(scalarsUShort, offset);
      fillEmptyConcentrationData(concArray, offset1);
   }


      removeCurrentActors();


   //Setting up contour extraction - notice this filter is allocated always but is also deleted every time it is allocated
   concContourFilter=vtkContourFilter::New();

   concContourFilter->SetInput(concVol);

   unsigned short currentContourNumber=0;
   for(set<unsigned short>::iterator sitr=distinctCellTypes.begin(); sitr!=distinctCellTypes.end() ;++sitr){
      cellType=*sitr;

      //filters got reallocated every time but they also get erased every time

      if ( ! univGraphSetPtr->avoidType(cellType) ){//only if cell types are not invisible      
//          cerr<<"currentContourNumber="<<currentContourNumber<<" cellType="<<cellType<<endl;
         concContourFilter->SetValue(currentContourNumber,cellType);
         ++currentContourNumber;

      }
   }

      UniversalGraphicsSettings::colorMapItr pos;
      QColor *colorPtr;
      double colorRGB[3];

      //setting backgroundColor to the color of Medium (type 0)
      pos=univGraphSetPtr->typeColorMap.find(0);
      if(pos!=univGraphSetPtr->typeColorMap.end()){
         colorPtr=&(pos->second);
      }else{
         colorPtr=&univGraphSetPtr->defaultColor;
      }
      QColorToVTKColor(*colorPtr, colorRGB);
      ren->SetBackground(colorRGB);

   
   //allocating actors and mappers and possibly other filters only when they have not been allocated before

   if(!concMapper){
      concMapper=vtkPolyDataMapper::New();
   }
   
   if(!concActor){
      concActor=vtkActor::New();
   }

   
//    cerr<<"minConcentrationTrue="<<minConcentrationTrue<<" maxConcentrationTrue="<<maxConcentrationTrue<<endl;


   if(!minConcentrationFixed){
      minConcentration = minConcentrationTrue;
   }

   if(!maxConcentrationFixed){
      maxConcentration = maxConcentrationTrue;
   }

   double concRangePlot[2];

   concRangePlot[0]=minConcentration;
   concRangePlot[1]=maxConcentration;
   
//    cerr<<"minConcentration="<<minConcentration<<" maxConcentration="<<maxConcentration<<endl;

   concMapper->SetInputConnection(concContourFilter->GetOutputPort());
   concMapper->ScalarVisibilityOn();
//    concMapper->SetScalarRange(&concRange[0]);
   concMapper->SetScalarRange(concRangePlot);
   concMapper->SetLookupTable(colorLookupTable);
   concMapper->SetScalarModeToUsePointFieldData();
   concMapper->ColorByArrayComponent("concentration",0);

   concActor->SetMapper(concMapper);
   



//    ren->Clear();
   ren->AddActor(concActor);

   if(legendEnable){
      legendActor->SetLookupTable(concMapper->GetLookupTable());

      legendActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
      legendActor->GetPositionCoordinate()->SetValue(0.1, 0.01);
      legendActor->SetOrientationToHorizontal();

      legendActor->SetWidth(0.8);
      legendActor->SetHeight(0.17);
      
      //setting up fontColor for legend to be in contrast with background . It will be changed 
      //in the future to allow users to pick the color...
      ren->GetBackground(colorRGB);
      colorRGB[0]>0.5 ? colorRGB[0]=0.0 : colorRGB[0]=1.0;
      colorRGB[1]>0.5 ? colorRGB[1]=0.0 : colorRGB[1]=1.0;
      colorRGB[2]>0.5 ? colorRGB[2]=0.0 : colorRGB[2]=1.0;
      legendActor->GetLabelTextProperty()->SetColor(colorRGB);
      ren->AddActor(legendActor);
   }

   


   //deallocating contour extraction filter
   concContourFilter->Delete();
   concContourFilter=0;

   if(!concMapper){
      concMapper->Delete();
      concMapper=0;
   }
   
   if(!concActor){
      concActor->Delete();
      concActor=0;
   }


   repaintVTK();
   setCameraRotations(data3DConf); //this has callsResetCamera which in turns re-renders the scene. It has to be called after repaint
//    cerr<<"AFTER REPAINT range_min="<<concRange[0]<<" range_max="<<concRange[1]<<endl;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool Display3DBase::checkIfInside(unsigned int x, unsigned y, unsigned z){

   //this function checks if a given pixed is fully surrounded by other non-medium pixels
   
//    static field3DGraphicsData_t & field3DGraphicsData = * field3DGraphicsDataPtr;
//    static unsigned int maxX=sizeL-1;
//    static unsigned int maxY=sizeM-1;
//    static unsigned int maxZ=sizeN-1;
//    
//    field3DGraphicsData= *field3DGraphicsDataPtr;
//    
//    if(x==0 || y==0 || z==0 || x==maxX||y==maxY||z==maxZ){
//       return false;
//    }else{
//       if(
//             0 != field3DGraphicsData[x-1][y][z].type &&
//             0 != field3DGraphicsData[x+1][y][z].type &&
//             0 != field3DGraphicsData[x][y-1][z].type &&
//             0 != field3DGraphicsData[x][y+1][z].type &&
//             0 != field3DGraphicsData[x][y][z-1].type &&
//             0 != field3DGraphicsData[x][y][z+1].type
//          )
//          //return false;
//          return true;
//       else
//          //return true;
//          return false;
//    }
   return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::prepareScene(){
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned int Display3DBase::legendDimension(std::string location, unsigned int &rectWidth,unsigned int & rectHeight,std::string type){

   return 0;  
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::paintLegend( float minConcentration, float maxConcentration,std::string location,std::string type){

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::prepareSceneColorMap(){

}


void Display3DBase::paint3DLattice(){

cerr<<"PAINT 3D Lattice"<<endl;


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::doBorders(unsigned int x,unsigned int y,unsigned int z ){

   
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Display3DBase::paint3DConcentrationLattice(){



}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::paintCellVectorFieldLattice(){
   //Need to fix assignment to arrays in the case when number of cells changes so that left over cells are not plotted
//    cerr<<"painting VECTOR FIELD"<<endl;
//    cerr<<"map size="<<currentVectorCellLevelFieldPtr->size()<<endl;
   GraphicsDataFields::vectorFieldCellLevel_t::iterator vitr;

   if(!drawingAllowed)
   return;

   unsigned int offset=0;
   float xCM,yCM,zCM;
   CompuCell3D::CellG* cell;
   Coordinates3D<float> vec;

   maxMagnitudeTrue=(numeric_limits<float>::min)();
   minMagnitudeTrue=(numeric_limits<float>::max)();
   float magnitude;


   //will need to either clean points
   for(vitr=currentVectorCellLevelFieldPtr->begin() ; vitr != currentVectorCellLevelFieldPtr->end() ; ++vitr){

//       cerr<<"cell.id="<<vitr->first->id<<" coordinate="<<vitr->second<<endl;
      cell=vitr->first;

      vec=Coordinates3D<float>(vitr->second);

      magnitude=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
      xCM=cell->xCM/(float)cell->volume;
      yCM=cell->yCM/(float)cell->volume;
      zCM=cell->zCM/(float)cell->volume;

      if(magnitude>maxMagnitudeTrue){
         maxMagnitudeTrue=magnitude;
      }

      if(magnitude<minMagnitudeTrue){
         minMagnitudeTrue=magnitude;
      }


      points->InsertPoint(offset,xCM,yCM,zCM);
      vectors->InsertTuple3(offset,vec.x,vec.y,vec.z);
//       cerr<<"xCM="<<xCM<<" yCM="<<xCM<<" zCM="<<zCM<<endl;
//       cerr<<"vec.x="<<vec.x<<" vec.y="<<vec.y<<" vec.z="<<vec.z<<" magnitude="<<magnitude<<endl;

      ++offset;

   }


   removeCurrentActors();

   

   UniversalGraphicsSettings::colorMapItr pos;
   QColor *colorPtr;
   double colorRGB[3];

   //setting backgroundColor to the color of Medium (type 0)
   pos=univGraphSetPtr->typeColorMap.find(0);
   if(pos!=univGraphSetPtr->typeColorMap.end()){
      colorPtr=&(pos->second);
   }else{
      colorPtr=&univGraphSetPtr->defaultColor;
   }
   QColorToVTKColor(*colorPtr, colorRGB);
   ren->SetBackground(colorRGB);


   if(!minMagnitudeFixed){
      minMagnitude = minMagnitudeTrue;
   }

   if(!maxMagnitudeFixed){
      maxMagnitude = maxMagnitudeTrue;
   }

//    minMagnitude=0;
//    maxMagnitude=60;

//    cerr<<" minMagnitude="<<minMagnitude<<" maxMagnitude="<<maxMagnitude<<endl;
//    cerr<<" minMagnitudeTrue="<<minMagnitudeTrue<<" maxMagnitudeTrue="<<maxMagnitudeTrue<<endl;

   double magnRangePlot[2];

   magnRangePlot[0]=minMagnitude;
   magnRangePlot[1]=maxMagnitude;

   

   if(!vectorMapper){
     vectorMapper=vtkPolyDataMapper::New();
   }

   if(!vectorActor){
      vectorActor=vtkActor::New();
   }


   if(!vectorGlyph){
      vectorGlyph=vtkGlyph3D::New();
   }

   if(!coneSource){
      coneSource=vtkConeSource::New();
      coneSource->SetResolution(5);
      coneSource->SetHeight(4);
      coneSource->SetRadius(1.5);

   }



//    cerr<<"vectorGlyph="<<vectorGlyph<<" coneSource="<<coneSource<<endl;
   

   vectorGlyph->SetInput(vectorGrid);
   vectorGlyph->SetSourceConnection(coneSource->GetOutputPort());
   //vectorGlyph->SetScaleModeToScaleByVector();
   vectorGlyph->SetColorModeToColorByVector();

// cerr<<"GOT PAST vectorGlyph->SetColorModeToColorByVector()"<<endl;
   
   vectorMapper->SetInputConnection(vectorGlyph->GetOutputPort());
//    vectorMapper->SetLookupTable(colorLookupTable);
   vectorMapper->SetScalarRange(magnRangePlot);
// vectorMapper->SetScalarRange(0,60);
   vectorActor->SetMapper(vectorMapper);
   
//    cerr<<"magnRangePlot[0]="<<magnRangePlot[0]<<" magnRangePlot[1]="<<magnRangePlot[1]<<endl;


//    ren->Clear();
   ren->AddActor(vectorActor);
   ren->AddActor(outlineActor);

   if(legendEnable){
      legendActor->SetLookupTable(vectorMapper->GetLookupTable());

      legendActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
      legendActor->GetPositionCoordinate()->SetValue(0.1, 0.01);
      legendActor->SetOrientationToHorizontal();

      legendActor->SetWidth(0.8);
      legendActor->SetHeight(0.17);
      
      //setting up fontColor for legend to be in contrast with background . It will be changed 
      //in the future to allow users to pick the color...
      ren->GetBackground(colorRGB);
      colorRGB[0]>0.5 ? colorRGB[0]=0.0 : colorRGB[0]=1.0;
      colorRGB[1]>0.5 ? colorRGB[1]=0.0 : colorRGB[1]=1.0;
      colorRGB[2]>0.5 ? colorRGB[2]=0.0 : colorRGB[2]=1.0;
      legendActor->GetLabelTextProperty()->SetColor(colorRGB);
      ren->AddActor(legendActor);
   }


   //or set points size here

   repaintVTK();
   setCameraRotations(data3DConf); //this has callsResetCamera which in turns re-renders the scene. It has to be called after 
   //deleting those objects is crucial otherwise you might get fake coloring of the glyphs as the old objects have "memory"
   vectorMapper->Delete();
   vectorMapper=0;
   vectorGlyph->Delete();
   vectorGlyph=0;
   vectorActor->Delete();
   vectorActor=0;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Display3DBase::paint3DCellVectorFieldLattice(){

//    cerr<<"painting VECTOR FIELD"<<endl;

}
