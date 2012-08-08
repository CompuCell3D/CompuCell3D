#ifndef COLORITEM_H
#define COLORITEM_H
#include <QtGui>

class ColorItem:public QTableWidgetItem{


   public:
      ColorItem(QTableWidget *_table, int row, int col, QColor & color);
      ~ColorItem();
      QColor &color();
      QColor &setColor(QColor &_color);
      
   private:
      
   
      QColor cellColor;
      QTableWidget *table;


};



#endif