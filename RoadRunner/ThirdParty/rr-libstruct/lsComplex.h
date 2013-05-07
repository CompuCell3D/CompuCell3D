#ifndef ls_COMPLEX_H
#define ls_COMPLEX_H

#include <iosfwd>
#include "lsExporter.h"

namespace ls
{

    /*! \class ls::Complex
        \brief ls::Complex is the complex class used by ls::LibLA

        This class implements a basic complex type along with basic operations on it.
    */

class LIB_EXTERN Complex
{
    public:
        //! real, imag part of the complex number
        double Real;
        double Imag;

    public:
        //! constructs a new complex number with given real and imaginary part
        Complex(double real = 0.0, double imag = 0.0);//: Real(real), Imag(imag) {}
        ////! virtual destructor
        ////virtual ~Complex() {}

        //! returns the real part of the complex number
        double getReal() const;

        //! return the complex part of the complex number
        double getImag() const;

        //! sets the real part of the complex number
        void setReal(double real);

        //! sets the imaginary part of the complex number
        void setImag(double imag);

        //! sets real and imaginary part of the complex number
        void set(double real, double imag);

        //! assignment operator (sets the real part only)
        Complex & operator = (const double rhs);

        //! assignment operator
        Complex & operator = (const Complex & rhs);

        //! implements addition of complex numbers
        Complex & operator + (const Complex & rhs);

        //! implements subtraction of complex numbers
        Complex & operator - (const Complex & rhs);

        //! implements multiplication of complex numbers
        Complex & operator * (const Complex & rhs);

        //! implements complex division
        Complex & operator / (const Complex & rhs);

        //! print the complex number on an output stream
        std::basic_ostream<char>& operator<<(std::basic_ostream<char>& os);
};

    /*! \brief overload that allows to print a complex number on a std::stream

        This function enables a complex number to be displayed on a stream. It will
        be formatted as: '(' + realpart + ' + ' + imaginaryPart + 'i)'. To use it
        invoke for example:
        \par
        Complex number(1.0, 0.5); cout << number << endl;

        \param os output stream to print on
        \param complex the complex number to be printed

        \return the output stream containing the printed complex number
    */

LIB_EXTERN double       real(const Complex& val);
LIB_EXTERN double       imag(const Complex& val);

LIB_EXTERN std::ostream& operator << (std::ostream &os,  const Complex & complex);

}
#endif
