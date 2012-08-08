#ifndef UNIVERSALGRAPHICSSETTINGS_H
#define UNIVERSALGRAPHICSSETTINGS_H
#include <map>
#include <vector>
#include <QColor>
#include <QPen>
#include <QBrush>

class UniversalGraphicsSettings{
   public:
      typedef std::map<unsigned short,QColor>::iterator colorMapItr;
      typedef std::map<unsigned short,QColor> colorMap_t;
      bool bordersOn;
      bool contoursOn;
      bool concentrationLimitsOn;
      int zoomFactor;
      QColor defaultColor;
      QColor borderColor;
      QColor contourColor;
      QColor arrowColor;
      std::vector<unsigned short> types3DInvisibleVec;
      //colormaps
      std::map<unsigned short,QPen> typePenMap;
      std::map<unsigned short,QBrush> typeBrushMap;
      std::map<unsigned short,QColor> typeColorMap;
      QBrush defaultBrush;
      QPen defaultPen;
      QPen borderPen;
      QPen contourPen;
      QPen arrowPen;
      bool avoidType(unsigned short type);

      
};

#endif
