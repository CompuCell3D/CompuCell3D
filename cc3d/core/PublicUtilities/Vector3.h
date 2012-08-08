#ifndef VECTOR3_H
#define VECTOR3_H
// This class is based on Vector3 from rook package

#include <iostream>


class Vector3{

public:

   typedef  double precision_t;
   Vector3(precision_t x = 0.0, precision_t y = 0.0, precision_t z = 0.0);
   // The constructor.

   Vector3(const Vector3 &);
   // The copy constructor.

   virtual ~Vector3();
   // Destructor

   precision_t operator () (int) const;
   inline precision_t operator [] (int) const;
   // Get components by index (Geant4).

   precision_t & operator () (int);
   inline precision_t & operator [] (int);
   // Set components by index.

   inline precision_t Perp2() const;
   precision_t Perp() const;
   inline precision_t Perp2(const Vector3 & p)  const ;

   //inline precision_t x()  const;
   //inline precision_t y()  const;
   //inline precision_t z()  const;
   //inline precision_t X()  const;
   //inline precision_t Y()  const;
   //inline precision_t Z()  const;
   //inline precision_t Px() const;
   //inline precision_t Py() const;
   //inline precision_t Pz() const;
   //// The components in cartesian coordinate system.

   //inline void SetX(precision_t);
   //inline void SetY(precision_t);
   //inline void SetZ(precision_t);
   inline void SetXYZ(precision_t x, precision_t y, precision_t z);
   // void        SetPtEtaPhi(precision_t pt, precision_t eta, precision_t phi);
   // void        SetPtThetaPhi(precision_t pt, precision_t theta, precision_t phi);
  

   precision_t Phi() const;
   // The azimuth angle. returns phi from -pi to pi 

   precision_t Theta() const;
   // The polar angle.

   inline precision_t CosTheta() const;
   // Cosine of the polar angle.

   inline precision_t Mag2() const;
   // The magnitude squared (rho^2 in spherical coordinate system).

   precision_t Mag() const;
   // The magnitude (rho in spherical coordinate system).

   void SetPhi(precision_t);
   // Set phi keeping mag and theta constant (BaBar).

   void SetTheta(precision_t);
   // Set theta keeping mag and phi constant (BaBar).

   inline void SetMag(precision_t);
   // Set magnitude keeping theta and phi constant (BaBar).
   
   void SetMagThetaPhi(precision_t mag, precision_t theta, precision_t phi);

   inline Vector3 & operator = (const Vector3 &);
   // Assignment.

   inline bool operator == (const Vector3 &) const;
   inline bool operator != (const Vector3 &) const;
   // Comparisons (Geant4).

   inline Vector3 & operator += (const Vector3 &);
   // Addition.

   inline Vector3 & operator -= (const Vector3 &);
   // Subtraction.

   inline Vector3 operator - () const;
   // Unary minus.

   inline Vector3 & operator *= (precision_t);
   // Scaling with real numbers.

   Vector3 Unit() const;
   // Unit vector parallel to this.

   inline Vector3 Orthogonal() const;
   // Vector orthogonal to this (Geant4).

   inline precision_t Dot(const Vector3 &) const;
   // Scalar product.

   inline Vector3 Cross(const Vector3 &) const;
   // Cross product.

   precision_t Angle(const Vector3 &) const;
   // The angle w.r.t. another 3-vector.



   void RotateX(precision_t);
   // Rotates the Hep3Vector around the x-axis.

   void RotateY(precision_t);
   // Rotates the Hep3Vector around the y-axis.

   void RotateZ(precision_t);
   // Rotates the Hep3Vector around the z-axis.

   void RotateUz(const Vector3&);
   // Rotates reference frame from Uz to newUz (unit vector) (Geant4).

   // void Rotate(precision_t, const Vector3 &);
   // Rotates around the axis specified by another Hep3Vector.


   //components
   precision_t fX, fY, fZ;




};


Vector3 operator + (const Vector3 &, const Vector3 &);
// Addition of 3-vectors.

Vector3 operator - (const Vector3 &, const Vector3 &);
// Subtraction of 3-vectors.

Vector3::precision_t operator * (const Vector3 &, const Vector3 &);
// Scalar product of 3-vectors.

Vector3 operator * (const Vector3 &, Vector3::precision_t a);
Vector3 operator * (Vector3::precision_t a, const Vector3 &);





Vector3::precision_t & Vector3::operator[] (int i)       { return operator()(i); }
Vector3::precision_t   Vector3::operator[] (int i) const { return operator()(i); }


inline void Vector3::SetXYZ(Vector3::precision_t xx, Vector3::precision_t yy, Vector3::precision_t zz) {
   fX = xx;
   fY = yy;
   fZ = zz;
}



inline Vector3 & Vector3::operator = (const Vector3 & p) {
   fX = p.fX;
   fY = p.fY;
   fZ = p.fZ;
   return *this;
}

inline bool Vector3::operator == (const Vector3& v) const {
   return (v.fX==fX && v.fY==fY && v.fZ==fZ) ? true : false;
}

inline bool Vector3::operator != (const Vector3& v) const {
   return (v.fX!=fX || v.fY!=fY || v.fZ!=fZ) ? true : false;
}

inline Vector3& Vector3::operator += (const Vector3 & p) {
   fX += p.fX;
   fY += p.fY;
   fZ += p.fZ;
   return *this;
}

inline Vector3& Vector3::operator -= (const Vector3 & p) {
   fX -= p.fX;
   fY -= p.fY;
   fZ -= p.fZ;
   return *this;
}

inline Vector3 Vector3::operator - () const {
   return Vector3(-fX, -fY, -fZ);
}

inline Vector3& Vector3::operator *= (Vector3::precision_t a) {
   fX *= a;
   fY *= a;
   fZ *= a;
   return *this;
}

inline Vector3::precision_t Vector3::Dot(const Vector3 & p) const {
   return fX*p.fX + fY*p.fY + fZ*p.fZ;
}

inline Vector3 Vector3::Cross(const Vector3 & p) const {
   return Vector3(fY*p.fZ-p.fY*fZ, fZ*p.fX-p.fZ*fX, fX*p.fY-p.fX*fY);
}

inline Vector3::precision_t Vector3::Mag2() const { return fX*fX + fY*fY + fZ*fZ; }


inline Vector3 Vector3::Orthogonal() const {
   precision_t xx = fX < 0.0 ? -fX : fX;
   precision_t yy = fY < 0.0 ? -fY : fY;
   precision_t zz = fZ < 0.0 ? -fZ : fZ;
   if (xx < yy) {
      return xx < zz ? Vector3(0,fZ,-fY) : Vector3(fY,-fX,0);
   } else {
      return yy < zz ? Vector3(-fZ,0,fX) : Vector3(fY,-fX,0);
   }
}



inline Vector3::precision_t Vector3::CosTheta() const {
   precision_t ptot = Mag();
   return ptot == 0.0 ? 1.0 : fZ/ptot;
}

inline void Vector3::SetMag(Vector3::precision_t ma) {
   precision_t factor = Mag();
   if (factor == 0) {
      return;
   } else {
      factor = ma/factor;
      fX=fX*factor;
      fY=fY*factor;
      fZ=fZ*factor;
   }
}


inline Vector3::precision_t Vector3::Perp2() const { return fX*fX + fY*fY; }




inline Vector3::precision_t Vector3::Perp2(const Vector3 & p)  const {
   precision_t tot = p.Mag2();
   precision_t ss  = Dot(p);
   precision_t per = Mag2();
   if (tot > 0.0) per -= ss*ss/tot;
   if (per < 0)   per = 0;
   return per;
}


inline std::ostream & operator<<(std::ostream & _out, const Vector3 &_a){
   _out<<"("<<_a.fX<<","<<_a.fY<<","<<_a.fZ<<")";
   return _out;
}





#endif