#ifndef _GRAPHICS2D_H
#define _GRAPHICS2D_H

#include "Graphics2DBase.h"
#include <QtGui>




class Graphics2D: public QLabel, public Graphics2DBase
{
   public:

      
      Graphics2D(QWidget *parent = 0, const char *name = 0);
      virtual ~Graphics2D();
      virtual void produceImage(QImage & image);
      virtual void preDrawTask();
      virtual void postDrawTask();
      virtual void initPainter();
      virtual void paintBackground(const QColor & _color);
                 

      QPixmap & getPixmap();


   private:

     QPixmap pixmap2D;
};

inline QPixmap & Graphics2D::getPixmap(){
   return pixmap2D;
}

#endif
