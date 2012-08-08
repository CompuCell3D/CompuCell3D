#include "ColorItem.h"

ColorItem::ColorItem(QTableWidget *_table, int row, int col, QColor & color):QTableWidgetItem(QString()),table(table),cellColor(color)
{
   QPixmap pixmap;
   pixmap=QPixmap(table->columnWidth(col),table->rowHeight(row));
// resize(table->columnWidth(col),table->rowHeight(row));
   pixmap.fill(color);
   setIcon(QIcon(pixmap));
}
ColorItem::~ColorItem(){};

QColor &ColorItem::color(){return cellColor;}

QColor &ColorItem::setColor(QColor &_color){cellColor=_color; return cellColor;}