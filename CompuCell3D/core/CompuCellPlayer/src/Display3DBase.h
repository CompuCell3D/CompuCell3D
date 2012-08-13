#ifndef DISPLAY3DBASE_H
#define DISPLAY3DBASE_H

// #include <QGLWidget>
// #include <qgl.h>
// #include <qpen.h>
#include <vector>
#include <CompuCell3D/Field3D/Point3D.h>
#include "GraphicsData.h"
#include <GraphicsDataFields.h>
#include "GraphicsBase.h"
#include "Configure3DData.h"



//Initially had problems with inherited signals because signal handlers were reeclared in Display3DBase...

// #include <qsplitter.h>

class QColor;
class Projection2DData;
class GraphicsDataFields;
class UniversalGraphicsSettings;

class vtkCylinderSource;
class vtkPolyDataMapper;
class vtkActor;
class vtkRenderer;
class vtkRenderWindow;
class vtkColorTransferFunction;
class vtkUnsignedShortArray;
class vtkPoints;
class vtkStructuredGrid;
class vtkStructuredPoints;
class vtkPiecewiseFunction;
class vtkColorTransferFunction;
class vtkVolumeProperty;
class vtkVolumeTextureMapper2D;
class vtkVolume;
class vtkMarchingCubes3D;
class vtkMarchingCubes;
class vtkPolyDataMapper;
class vtkDiscreteMarchingCubes ;
class vtkSmoothPolyDataFilter;
class vtkPolyDataNormals;
class vtkContourFilter;
class vtkFloatArray ;
class vtkLookupTable;
class vtkScalarBarActor ;
class vtkWindowToImageFilter;
class vtkPNGWriter;
class vtkCamera;
class vtkUnstructuredGrid;
class vtkGlyph3D;
class vtkConeSource ;
class vtkOutlineFilter;




class Display3DBase : public GraphicsBase
{

public:
   typedef GraphicsDataFields::field3DGraphicsData_t field3DGraphicsData_t;
   typedef GraphicsDataFields::floatField3D_t floatField3D_t;
   typedef float GLfloat;

    Display3DBase(const char *name = 0);
   virtual ~Display3DBase();
   virtual void paintLattice();
   virtual void paintLegend( float minConcentration, float maxConcentration,std::string location,std::string type);
   virtual unsigned int legendDimension(std::string location, unsigned int &rectWidth,unsigned int & rectHeight,std::string type);
   virtual void paintConcentrationLattice();
   virtual void fillFakeConcentration(){};
   virtual void paintCellVectorFieldLattice();
   virtual void doContourLines(){};
   virtual void produceImage(QImage & image){};
   virtual void produceImage(const std::string & _fileName);
   virtual void paintGLCall(){paintGL();}
   virtual void repaintVTK();
   virtual void initializeDisplay3D();
   
   void initializeVTKSettings();     

   float getRotationX();
   float getRotationY();
   float getRotationZ();
    
    void setSizeLMN(unsigned int _sizeL , unsigned int _sizeM ,unsigned int _sizeN);
    void setDrawingAllowedFlag(bool);
    
    Configure3DData getConfigure3DData();
    void setConfigure3DData(const Configure3DData & _data);
    void setInitialConfigure3DData(const Configure3DData & _data);

   
    
protected:

    void (Display3DBase::*drawFcnPtr)();
    //bool avoidType(unsigned short type);
    void setDimensions(GLfloat _sizeX,GLfloat _sizeY, GLfloat _sizeZ );
    void drawCube(GLfloat x, GLfloat y , GLfloat z , GLfloat side );
    void drawCube(GLfloat x, GLfloat y , GLfloat z , GLfloat side, const QColor &color );
    void drawSquare(GLfloat x, GLfloat y , GLfloat z , GLfloat side, const QColor &color);
    void drawRectangle(GLfloat x, GLfloat y , GLfloat z , GLfloat width , GLfloat height, const QColor &color);
    void initDraw();
    void prepareScene();//pushes matrix
    void prepareSceneColorMap();//pushes matrix
    void paint3DLattice();//must pop matrixv
    void paint3DConcentrationLattice();//must pop matrixv
    void paint3DCellVectorFieldLattice();//must pop matrixv
    
    void initializeGL();
    void resizeGL(int width, int height);
    
    void paintGL();


    void draw();
    int faceAtPosition(const QPoint &pos);
    bool checkIfInside(unsigned int x, unsigned y, unsigned z);
    void doBorders(unsigned int x,unsigned int y,unsigned int z );
    bool belongsToField(CompuCell3D::Point3D &pt);
    void QColorToVTKColor(const QColor & color, double * _vtkColorArray); 
    void setCameraRotations(const  Configure3DData &_data3DConf);
    void removeCurrentActors();
    void initializeDataSetSize(vtkStructuredPoints * dataSet, int xDim,int yDim,int zDim );
    void fillEmptyCellTypeData(vtkUnsignedShortArray *_scalarsUShort, unsigned int & _offset);
    void fillEmptyConcentrationData(vtkFloatArray *_concArray, unsigned int & _offset);
    GLfloat rotationX;
    GLfloat rotationY;
    GLfloat rotationZ;
    QColor faceColors[6];
    QPoint lastPos;
    
    std::vector<std::vector<std::vector<GLfloat> > >   cubeCoords;
    std::vector< GLfloat >   cubeCoordsVec;
    GLfloat sizeX,sizeY,sizeZ;
    
    Configure3DData data3DConf;
    Configure3DData mostRecent3DConf;

    
    unsigned int sizeL,sizeM,sizeN;
    unsigned int maxDimension;

    
    field3DGraphicsData_t * field3DGraphicsDataPtr;
    
    unsigned int currentZoomFactor;
    bool drawingAllowed;
    unsigned int legendOffset;
    float enlargementFactor;

     std::vector<vtkActor*> typeActorsVec;
     std::vector<vtkSmoothPolyDataFilter*> smootherFilterVec;
     std::vector<vtkPolyDataNormals*> polyDataNormalsVec;
     std::vector<vtkDiscreteMarchingCubes*> typeExtractorVec;
//      std::vector<vtkContourFilter*> typeExtractorVec;     
     std::vector<vtkPolyDataMapper*> typeExtractorMapperVec;


     vtkContourFilter * concContourFilter;
     vtkPolyDataMapper* concMapper;
     vtkActor* concActor;
     vtkStructuredPoints *concVol;
     vtkFloatArray *concArray;
     vtkLookupTable *colorLookupTable; 
     vtkScalarBarActor *legendActor;

   


     vtkRenderer* ren;
     vtkRenderWindow * renwin;

     vtkWindowToImageFilter *wimFilter;
     vtkPNGWriter *pngWriter;
     int dims[3];
     
     vtkUnsignedShortArray *scalarsUShort;
     

     vtkPoints *points;
     vtkStructuredGrid *sgrid;
     vtkStructuredPoints *vol;
     vtkUnstructuredGrid *vectorGrid;     
     vtkFloatArray *vectors;
     vtkPolyDataMapper* vectorMapper;
     vtkGlyph3D *vectorGlyph;
     vtkConeSource *coneSource;
     vtkActor *vectorActor;
     
    //lattice box
    vtkOutlineFilter *outlineFilter;
    vtkPolyDataMapper *outlineMapper;
    vtkActor *outlineActor;

     vtkCamera *camera;
     vtkCamera *activeCamera;


    bool initialized;

};

inline bool Display3DBase::belongsToField(CompuCell3D::Point3D &pt){
   if(pt.x>=0 && pt.x<sizeL && pt.y>=0 && pt.y<sizeM && pt.z>=0 && pt.z<sizeN)
      return true;
   else
      return false;
}



inline float Display3DBase::getRotationX(){return rotationX;}
inline float Display3DBase::getRotationY(){return rotationY;}
inline float Display3DBase::getRotationZ(){return rotationZ;}
inline void Display3DBase::setDrawingAllowedFlag(bool _flag){drawingAllowed=_flag;}

inline Configure3DData Display3DBase::getConfigure3DData(){return data3DConf;}
inline void Display3DBase::setConfigure3DData(const Configure3DData & _data){data3DConf=_data;}





#endif
