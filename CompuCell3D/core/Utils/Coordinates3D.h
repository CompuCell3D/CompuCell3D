#ifndef COORDINATES3D_H
#define COORDINATES3D_H
#include <iostream>

///T must provide copy constructor and operator=
template <typename T>
class Coordinates3D{

   public:
      Coordinates3D():
                     x(T()),
                     y(T()),
                     z(T())
                     
      {}
      
      Coordinates3D(const T & _x,const T & _y,const T & _z){
         x=_x;
         y=_y;
         z=_z;
      }

      T  X() const {return x;}
      T  Y() const {return y;}
      T  Z() const {return z;}

      T&  XRef(){return x;}
      T&  YRef(){return y;}
      T&  ZRef(){return z;}

      Coordinates3D<T> & operator-=(const Coordinates3D<T> &a){
         x-=a.X();
         y-=a.Y();
         z-=a.Z();
         return *this;
      }
      Coordinates3D<T> & operator+=(const Coordinates3D<T> &a){
         x+=a.X();
         y+=a.Y();
         z+=a.Z();
         return *this;
      }

      Coordinates3D<T> & operator=(const Coordinates3D<T> &a){
         x=a.X();
         y=a.Y();
         z=a.Z();
         return *this;
      }
      

   
      T x;
      T y;
      T z;

};


template<typename T>
Coordinates3D<T> crossProduct(const Coordinates3D<T> &a,const Coordinates3D<T> &b){
   return Coordinates3D<T>(a.y*b.z-a.z*b.y , a.z*b.x-a.x*b.z , a.x*b.y-a.y*b.x);
}


template<typename T>
Coordinates3D<T> operator+(const Coordinates3D<T> &a,const Coordinates3D<T> &b){
   return Coordinates3D<T>(a.X()+b.X(),a.Y()+b.Y(),a.Z()+b.Z());
}

template<typename T>
Coordinates3D<T> operator-(const Coordinates3D<T> &a,const Coordinates3D<T> &b){
   return Coordinates3D<T>(a.X()-b.X(),a.Y()-b.Y(),a.Z()-b.Z());
}

template<typename T> //scalar product
T operator*(const Coordinates3D<T> &a,const Coordinates3D<T> &b){
   return a.x*b.x+a.y*b.y+a.z*b.z;
}
   
template<typename T>
std::ostream & operator<<(std::ostream &_out,Coordinates3D<T> & _coordinates){

      _out<<_coordinates.X()<<" , "<<_coordinates.Y()<<" , "<<_coordinates.Z();
	  return _out;

}

template<typename T>
Coordinates3D<T> operator<(const Coordinates3D<T> &a,const Coordinates3D<T> &b){
   return a.X()<b.X() || (!(a.X()<b.X()) && a.Y()< b.Y()  ) || (!(a.X()<b.X()) && !(a.Y()< b.Y()) &&  a.Z()< b.Z());
}


template<typename T>
std::ostream & operator<<(std::ostream & _out, const Coordinates3D<T> &_a){
   _out<<"("<<_a.X()<<","<<_a.Y()<<","<<_a.Z()<<")";
   return _out;
}



#endif
