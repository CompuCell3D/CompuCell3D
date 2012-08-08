#ifndef _GRAPHICS2DBASE_H
#define _GRAPHICS2DBASE_H

#include "GraphicsBase.h"
#include "Projection2DData.h"
#include <QtGui>

// #include <qlabel.h>
// #include <qpixmap.h>

class QPainter;

template <typename T> 
class Coordinates3D;

class Graphics2DBase:  public GraphicsBase{

   public:

      
      Graphics2DBase( const char *name = 0);
      virtual ~Graphics2DBase();
                 
      virtual void paintLattice();
      virtual void paintLegend( float minConcentration, float maxConcentration,std::string location,std::string type);
      virtual unsigned int legendDimension(std::string location, unsigned int &rectWidth,unsigned int & rectHeight,std::string type);
      virtual void paintConcentrationLattice();
      virtual void paintCellVectorFieldLattice();
//       virtual void fillFakeConcentration();
      virtual void doContourLines();
      virtual void produceImage(QImage & image);

      virtual void preDrawTask();
      virtual void postDrawTask();
      virtual void initPainter();
      virtual void paintBackground(const QColor & _color);


//       QPixmap & getPixmap();
//       QImage & getImage();
      Projection2DData projData;
            
   protected:

      Coordinates3D<double> HexCoordXY(unsigned int x,unsigned int y,unsigned int z);
      void drawHexagonXY(int x, int y, int scaleM,  QPainter *painter);
      void drawRectangleXY(int x, int y, int scaleM,  QPainter *painter);
      void paintHexPixelXY(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      void paintHexBordersXY(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);

      void paintCellVectorFieldLatticeCore(bool _erasePrevious=true);
      void paintLatticeCore(bool _erasePrevious=true);
      QImage::Format imageFormat;
      ///QT graphics  methods
      void paintBordersXY(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      void paintPixelXY(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      
      void paintBordersXZ(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      void paintPixelXZ(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      
      void paintBordersYZ(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);
      void paintPixelYZ(int l,int m, int n,int scaleM, int scaleN,QPainter *painter);

      void drawArrow(int x, int y ,float vX, float vY , int scaleM, int scaleN ,int arrowLength, QPainter *painter, double _lengthScale=1.0);
      
      ///Cell Border Detector

      bool rightBorderXY( int _L, int _M, int _N);
      bool leftBorderXY(  int _L, int _M, int _N);
      bool upperBorderXY(  int _L, int _M,  int _N);
      bool bottomBorderXY(  int _L, int _M, int _N);

      bool rightBorderXZ( int _L, int _M, int _N);
      bool leftBorderXZ(  int _L, int _M, int _N);
      bool upperBorderXZ(  int _L, int _M,  int _N);
      bool bottomBorderXZ(  int _L, int _M, int _N);

      bool rightBorderYZ(int _L, int _M, int _N);
      bool leftBorderYZ(int _L, int _M, int _N);
      bool upperBorderYZ(int _L, int _M,  int _N);
      bool bottomBorderYZ(int _L, int _M, int _N);
      QSize labelSize;
//       QPixmap pixmap2D;
//       QImage image2D;
      QPainter *painterPtr;
      QPolygonF hexagon;
      QVector<QLineF> bordersHex;
      
       
};

// inline QImage & Graphics2DBase::getImage(){
//    return image2D;
// }

#endif
