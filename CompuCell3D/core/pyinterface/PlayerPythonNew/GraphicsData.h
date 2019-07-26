#ifndef _GRAPHICSDATA_H
#define _GRAPHICSDATA_H

class GraphicsData{

   public:
      GraphicsData():type(0),id(0),flag(0),averageConcentration(0)
      {}
      unsigned short type;
      long id;
      unsigned char flag;
      float averageConcentration;

};



#endif
