#include <cmath>

#include "Vector3.h"

//______________________________________________________________________________
Vector3::Vector3(const Vector3 & p):
  fX(p.fX), fY(p.fY), fZ(p.fZ) {}

Vector3::Vector3(precision_t xx, precision_t yy, precision_t zz)
: fX(xx), fY(yy), fZ(zz) {}

Vector3::~Vector3() {}

//______________________________________________________________________________
Vector3::precision_t Vector3::operator () (int i) const {
   //dereferencing operator const
   switch(i) {
      case 0:
         return fX;
      case 1:
         return fY;
      case 2:
         return fZ;
      default:
        return fX;
   }
   return 0.;
}

//______________________________________________________________________________
Vector3::precision_t & Vector3::operator () (int i) {
   //dereferencing operator
   switch(i) {
      case 0:
         return fX;
      case 1:
         return fY;
      case 2:
         return fZ;
      default:
         return fX;
   }
   return fX;
}

//______________________________________________________________________________
Vector3::precision_t Vector3::Perp() const{
	return sqrt(Perp2()); 
}


//______________________________________________________________________________
Vector3::precision_t Vector3::Angle(const Vector3 & q) const 
{
   // return the angle w.r.t. another 3-vector
   precision_t ptot2 = Mag2()*q.Mag2();
   if(ptot2 <= 0) {
      return 0.0;
   } else {
      precision_t arg = Dot(q)/sqrt(ptot2);
      if(arg >  1.0) arg =  1.0;
      if(arg < -1.0) arg = -1.0;
      return acos(arg);
   }
}

//______________________________________________________________________________
Vector3::precision_t Vector3::Mag() const 
{ 
   // return the magnitude (rho in spherical coordinate system)
   
   return sqrt(Mag2()); 
}

//______________________________________________________________________________
Vector3::precision_t Vector3::Phi() const 
{
   //return the  azimuth angle. returns phi from -pi to pi
   return fX == 0.0 && fY == 0.0 ? 0.0 : atan2(fY,fX);
}

//______________________________________________________________________________
Vector3::precision_t Vector3::Theta() const 
{
   //return the polar angle
   return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : atan2(Perp(),fZ);
}

//______________________________________________________________________________
Vector3 Vector3::Unit() const 
{
   // return unit vector parallel to this.
   precision_t  tot = Mag2();
   Vector3 p(fX,fY,fZ);
   return tot > 0.0 ? p *= (1.0/sqrt(tot)) : p;
}

//______________________________________________________________________________
void Vector3::RotateX(precision_t angle) {
   //rotate vector around X
   precision_t s = sin(angle);
   precision_t c = cos(angle);
   precision_t yy = fY;
   fY = c*yy - s*fZ;
   fZ = s*yy + c*fZ;
}

//______________________________________________________________________________
void Vector3::RotateY(precision_t angle) {
   //rotate vector around Y
   precision_t s = sin(angle);
   precision_t c = cos(angle);
   precision_t zz = fZ;
   fZ = c*zz - s*fX;
   fX = s*zz + c*fX;
}

//______________________________________________________________________________
void Vector3::RotateZ(precision_t angle) {
   //rotate vector around Z
   precision_t s = sin(angle);
   precision_t c = cos(angle);
   precision_t xx = fX;
   fX = c*xx - s*fY;
   fY = s*xx + c*fY;
}

//______________________________________________________________________________
void Vector3::RotateUz(const Vector3& NewUzVector) {
   // NewUzVector must be normalized !

   precision_t u1 = NewUzVector.fX;
   precision_t u2 = NewUzVector.fY;
   precision_t u3 = NewUzVector.fZ;
   precision_t up = u1*u1 + u2*u2;

   if (up) {
      up = sqrt(up);
      precision_t px = fX,  py = fY,  pz = fZ;
      fX = (u1*u3*px - u2*py + u1*up*pz)/up;
      fY = (u2*u3*px + u1*py + u2*up*pz)/up;
      fZ = (u3*u3*px -    px + u3*up*pz)/up;
   } else if (u3 < 0.) { fX = -fX; fZ = -fZ; }      // phi=0  teta=pi
   else {};
}

//______________________________________________________________________________
void Vector3::SetTheta(precision_t th) 
{
   // Set theta keeping mag and phi constant (BaBar).
   precision_t ma   = Mag();
   precision_t ph   = Phi();
   fX=ma*sin(th)*cos(ph);
   fY=ma*sin(th)*sin(ph);
   fZ=ma*cos(th);
}

//______________________________________________________________________________
void Vector3::SetPhi(precision_t ph) 
{
   // Set phi keeping mag and theta constant (BaBar).
   precision_t xy   = Perp();
   fX=xy*cos(ph);
   fY=xy*sin(ph);
}

//______________________________________________________________________________
void Vector3::SetMagThetaPhi(precision_t mag, precision_t theta, precision_t phi) 
{
   //setter with mag, theta, phi
   precision_t amag = fabs((float)mag);
   fX = amag * sin(theta) * cos(phi);
   fY = amag * sin(theta) * sin(phi);
   fZ = amag * cos(theta);
}


Vector3 operator + (const Vector3 & a, const Vector3 & b) {
   return Vector3(a.fX + b.fX, a.fY + b.fY, a.fZ + b.fZ);
}

Vector3 operator - (const Vector3 & a, const Vector3 & b) {
   return Vector3(a.fX - b.fX, a.fY - b.fY, a.fZ - b.fZ);
}

Vector3 operator * (const Vector3 & p, Vector3::precision_t a) {
   return Vector3(a*p.fX, a*p.fY, a*p.fZ);
}

Vector3 operator * (Vector3::precision_t a, const Vector3 & p) {
   return Vector3(a*p.fX, a*p.fY, a*p.fZ);
}

Vector3::precision_t operator * (const Vector3 & a, const Vector3 & b) {
   return a.Dot(b);
}



