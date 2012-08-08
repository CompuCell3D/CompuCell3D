#ifndef PROJECTION2DDATA_H
#define PROJECTION2DDATA_H
#include <string>
#include <iostream>

class QLabel;
class QPainter;

class Projection2DData{

   public:
      Projection2DData():
      projection("none"),
      sizeL(0),sizeM(0),
      xMin(0),xMax(0),
      yMin(0),yMax(0),
      zMin(0),zMax(0)
/*      imageLabelPtr(0),
      painterPtr(0)*/
      
      {}
      void resetBoundaries(unsigned int _xMax, unsigned int _yMax, unsigned int _zMax){
         xMin=yMin=zMin=0;
         xMax=_xMax;
         yMax=_yMax;
         zMax=_zMax;
      }
      bool checkIfCompatibleWithField(unsigned int _sizeL,unsigned int _sizeM,unsigned int _sizeN) const;
      std::string projection;
      unsigned int sizeL,sizeM;
      unsigned int xMin,xMax;
      unsigned int yMin,yMax;
      unsigned int zMin,zMax;      

};

inline std::ostream & operator<<(std::ostream & out,const Projection2DData & projData){
   using namespace std;
   out<<projData.projection<<" ";
   out<<projData.sizeL<<" ";
   out<<projData.sizeM<<" ";
   out<<projData.xMin<<" ";
   out<<projData.xMax<<" ";
   out<<projData.yMin<<" ";
   out<<projData.yMax<<" ";
   out<<projData.zMin<<" ";
   out<<projData.zMax<<" ";
   return out;
   

}

inline std::istream & operator>>(std::istream & in, Projection2DData & projData){
   using namespace std;
   in>>projData.projection;
   in>>projData.sizeL;
   in>>projData.sizeM;
   in>>projData.xMin;
   in>>projData.xMax;
   in>>projData.yMin;
   in>>projData.yMax;
   in>>projData.zMin;
   in>>projData.zMax;
   return in;

}

inline bool Projection2DData::checkIfCompatibleWithField(unsigned int _sizeL,unsigned int _sizeM,unsigned int _sizeN) const{

   if(projection=="xy"){
      if(sizeL != _sizeL || sizeM != _sizeM || xMax>_sizeL || yMax>_sizeM || zMax>_sizeN)
         return false;
      else
         return true;
      
   }
   else if(projection=="xz"){
      if(sizeL != _sizeL || sizeM != _sizeN || xMax>_sizeL|| yMax>_sizeM || zMax>_sizeN)
         return false;
      else
         return true;
      
   }
   else if(projection=="yz"){
      if(sizeL != _sizeM || sizeM != _sizeN || xMax>_sizeL|| yMax>_sizeM || zMax>_sizeN)
         return false;
      else
         return true;
      
   }   
}

#endif
