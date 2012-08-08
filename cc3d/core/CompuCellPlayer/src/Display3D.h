#ifndef DISPLAY3D_H
#define DISPLAY3D_H


#include <QVTKWidget.h>
// #include <QtGui>
#include <Display3DBase.h>





class Display3D : public QVTKWidget,public Display3DBase
{

public:

    Display3D(QWidget *parent = 0, const char *name = 0);
    virtual ~Display3D();
    virtual void initializeDisplay3D();


   QVTKWidget *getQVTKWidget();

//   signals:
//     // Description:
//     // This signal will be emitted whenever a mouse event occurs
//     // within the QVTK window
//     void mouseEvent(QMouseEvent* event);

    
protected:

/*    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);*/
    

    protected:
//       virtual void mouseReleaseEvent(QMouseEvent* event);
    
private:
     QVTKWidget *qvtkWidget;
     
};


#endif
