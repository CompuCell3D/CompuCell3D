#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <ostream>
#include "lsComplex.h"

using namespace std;

namespace ls
{

Complex::Complex(double real, double imag) :
    Real(real),
    Imag(imag)
{}


//! returns the real part of the complex number
double Complex::getReal() const { return Real; }

//! return the complex part of the complex number
double Complex::getImag() const { return Imag; }

//! sets the real part of the complex number
void Complex::setReal(double real) { Real = real; }

//! sets the imaginary part of the complex number
void Complex::setImag(double imag) { Imag = imag; }

//! sets real and imaginary part of the complex number
void Complex::set(double real, double imag) { setReal(real); setImag(imag);}

//! assignment operator (sets the real part only)
Complex& Complex::operator = (const double rhs)
{
    Real = rhs;
    return *this;
}

//! assignment operator
Complex& Complex::operator = (const Complex & rhs)
{
    Real = rhs.Real;
    Imag = rhs.Imag;
    return *this;
}

//! implements addition of complex numbers
Complex& Complex::operator + (const Complex & rhs)
{
    Real += rhs.Real;
    Imag += rhs.Imag;
    return *this;
}

//! implements subtraction of complex numbers
Complex& Complex::operator - (const Complex & rhs)
{
    Real -= rhs.Real;
    Imag -= rhs.Imag;
    return *this;
}

//! implements multiplication of complex numbers
Complex& Complex::operator * (const Complex & rhs)
{
    Real = Real * rhs.Real - Imag*rhs.Imag;
    Imag = Imag*rhs.Real + Real*rhs.Imag;
    return *this;
}

//! implements complex division
Complex& Complex::operator / (const Complex & rhs)
{
    Real = (Real * rhs.Real + Imag*rhs.Imag)/(rhs.Real*rhs.Real+rhs.Imag*rhs.Imag);
    Imag = (Imag*rhs.Real - Real*rhs.Imag)/(rhs.Real*rhs.Real+rhs.Imag*rhs.Imag);
    return *this;
}


//Non class utility functions..
double real(const Complex& val)
{
    return val.Real;
}

double imag(const Complex& val)
{
    return val.Imag;
}

std::basic_ostream<char>& Complex::operator<<(std::basic_ostream<char>& os)
{
    return os << "(" << Real << " + " << Imag << "i)";
}

ostream& operator << (ostream& os,  const Complex& cpx)
{
    return os << "(" << cpx.Real << " + " << cpx.Imag << "i)";
}

}
