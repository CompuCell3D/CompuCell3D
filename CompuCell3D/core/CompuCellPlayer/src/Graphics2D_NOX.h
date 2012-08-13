#ifndef _GRAPHICS2D_NOX_H
#define _GRAPHICS2D_NOX_H

#include "Graphics2DBase.h"

#include <QtGui>

class QPainter;


class Graphics2D_NOX:  public Graphics2DBase{

   public:

      
      Graphics2D_NOX( const char *name = 0);
      virtual ~Graphics2D_NOX();
      virtual void produceImage(QImage & image);
      virtual void preDrawTask();
      virtual void postDrawTask();
      virtual void initPainter();
      virtual void paintBackground(const QColor & _color);
                 
      QImage & getImage();

   private:

      QImage image2D;

      
       
};

inline QImage & Graphics2D_NOX::getImage(){
   return image2D;
}

#endif
