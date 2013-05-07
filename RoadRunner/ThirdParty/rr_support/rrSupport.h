#ifndef rrSupportH
#define rrSupportH
#include "rrCExporter.h"

#ifndef MATH_CONSTANTS
#define E 2.71828182845904523536028747135
#define PI 3.14159265358979323846264338327
#endif

// Boolean functions for event handling" + NL());
double spf_gt(double a, double b);
double spf_lt(double a, double b);
double spf_geq(double a, double b);
double spf_leq(double a, double b);
double spf_eq(double a, double b);
double spf_neq(double a, double b);
double spf_and(int numArgs, ...);
double spf_or(int numArgs, ...);
double spf_xor(int numArgs, ...);

//D_S double     spf_not(double a);
//D_S bool     spf_not(bool a);
//D_S double     spf_xor(params double[] a);
//D_S bool     spf_xor(params bool[] a);
double spf_abs(double s);
double spf_exp(double a);
double spf_pow(double a, double b);
double spf_ceil(double a);
double spf_floor(double a);
int    spf_factorial(int a);
double spf_log(double a);
double spf_log10(double a);

//double spf_log(double a, double b);
double spf_delay(double a, double b);
double spf_root(double a, double b);

//double spf_piecewise(params object[] args);
double spf_piecewise(int nrOfArgs, ...);

// Square
double spf_sqr(double a);
double Logbase(double value, double baseValue);

//// -----------------------------------------------------------------------
//// Start of trig functions
//// -----------------------------------------------------------------------

//// Convert degrees to Radians
double degToRad(double degrees);

// Convert radians to degrees
double radToDeg(double radians);


double spf_sin(double a);

//// Secant
double sec(double a);

//// Cotangent
double cot(double a);
//
//// Inverse cotangent
double arccot(double a);
//
//// Inverse cotangent - ratio numerator and denominator provided
double arccot2(double a, double b);
//// Inverse secant
double asec(double a);

// Inverse secant
double arcsech(double a);

//// Cosecant
double csc(double a);

//// Inverse cosecant
double acsc(double a);

//// Hyperbolic secant of a double number
double sech(double a);

//// Inverse hyperbolic secant of a double number
double asech(double a);

//// Hyperbolic cosecant of a double number
double csch(double a);

//// Inverse hyperbolic cosecant of a double number
double arccsc(double a);

//// Inverse hyperbolic cosecant of a double number
double arccsch(double a);

//// Hyperbolic cotangent of a double number
double coth(double a);

//// Inverse hyperbolic cotangent of a double number
double arccoth(double a);

//// Inverse hyperbolic functions
//// --------------------------------------------------------------
//// Inverse hyperbolic sine of a double number
double arcsinh(double a);

//// Inverse hyperbolic cosine of a double number
double arccosh(double a);

//// Inverse hyperbolic tangent of a double number
double arctanh(double a);

//// Boolean functions for event handling" + NL());
//D_S double Gt(double a, double b);
//D_S double Lt(double a, double b);
//D_S double Geq(double a, double b);
//D_S double Leq(double a, double b);
//D_S double Eq(double a, double b);
//D_S double Neq(double a, double b);
//D_S double And(double first, ...);    //double args...
////D_S bool And(params bool[] a);
////D_S double Or(params double[] a);
////D_S bool Or(params bool[] a);
//D_S double Not(double a);
////D_S bool Not(bool a);
////D_S double Xor(params double[] a);
////D_S bool Xor(params bool[]  a);
//
////No references to 'double' Factorial
//D_S double Factorial(double a);
////D_S double log(double a);
////D_S double log(double a, double b);
//D_S double Delay(double a, double b);
//D_S double Root(double a, double b);
////D_S double Piecewise(params object[] args);
//


// See: http://en.wikipedia.org/wiki/Mathematical_constant
//const double EULER_CONSTANT_GAMMA  = 0.57721566490153286060651209008;
//const double GOLDEN_RATIO          = 1.618033988749895;
//const double LOG2E                 = 1.44269504088896340735992468100; /* log_2 (e) */
//const double LOG10E                = 0.43429448190325182765112891892; /* log_10 (e) */
//const double SQRT2                 = 1.41421356237309504880168872421; /* sqrt(2) */
//const double SQRT1_2               = 0.70710678118654752440084436210; /* sqrt(1/2) */
//const double SQRT3                 = 1.73205080756887729352744634151; /* sqrt(3) */
//const double PI_BY_2               = 1.57079632679489661923132169164; /* pi/2 */
//const double PI_BY_4               = 0.78539816339744830966156608458; /* pi/4 */
//const double SQRTPI                = 1.77245385090551602729816748334; /* sqrt(pi) */
//const double TWO_BY_SQRTPI         = 1.12837916709551257389615890312; /* 2/sqrt(pi) */
//const double ONE_BY_PI             = 0.31830988618379067153776752675; /* 1/pi */
//const double TWO_BY_PI             = 0.63661977236758134307553505349; /* 2/pi */
//const double LN10                  = 2.30258509299404568401799145468; /* ln(10) */
//const double LN2                   = 0.69314718055994530941723212146; /* ln(2) */
//const double LNPI                  = 1.14472988584940017414342735135; /* ln(pi) */
#endif


