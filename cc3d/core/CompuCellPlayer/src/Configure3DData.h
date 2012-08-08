#ifndef CONFIGURE3DDATA_H
#define CONFIGURE3DDATA_H
#include <iostream>

class Configure3DData{
   public:
      Configure3DData():sizeX(0),sizeY(0),sizeZ(0),rotationX(0),rotationY(0),rotationZ(0)
      {}
      ~Configure3DData(){}
      unsigned int sizeX, sizeY,sizeZ;
      int rotationX,rotationY,rotationZ;
      unsigned int sizeL, sizeM, sizeN;
      
      

};

inline std::ostream & operator<<(std::ostream & out,const Configure3DData & data3D){

   out<<data3D.sizeX<<" ";
   out<<data3D.sizeY<<" ";
   out<<data3D.sizeZ<<" ";
   out<<data3D.rotationX<<" ";
   out<<data3D.rotationY<<" ";
   out<<data3D.rotationZ<<" ";
   return out;

}

inline std::istream & operator>>(std::istream & in, Configure3DData & data3D){
//    int l;
//    
//    in>>l;
//    data3D.sizeX=l;
//    in>>l;
//    data3D.sizeY=l;
//    in>>l;
//    data3D.sizeZ=l;
   
   in>>data3D.sizeX;
   in>>data3D.sizeY;
   in>>data3D.sizeZ;
   in>>data3D.rotationX;
   in>>data3D.rotationY;
   in>>data3D.rotationZ;
   
/*   in>>data3D.sizeY;
   in>>data3D.sizeZ;*/
   return in;

}



#endif