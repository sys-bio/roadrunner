/*
 * SBMLSupportFunctions.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "SBMLSupportFunctions.h"

#define _USE_MATH_DEFINES
#include <cmath>

namespace sbmlsupport
{
extern "C" {
double trunc(double d)
{
    return (d > 0) ? floor(d) : ceil(d) ;
}

double poweri(double x, int y)
{
//The following code goes into an endless loop when y is negative
//    double value = 1.0;
//    int count = 0;
//    do
//    {
//        if(IS_ODD(exponent))
//        {
//            value *= base;
//        }
//        exponent >>= 1;
//        base *= base;
//        ++count;
//    }
//    while (exponent);
//    return (value);

    double  temp;
    if( y == 0)
    {
       return 1;
    }

    temp = poweri(x, y/2);
    if (y%2 == 0)
    {
        return temp*temp;
    }
    else
    {
        if(y > 0)
            return x*temp*temp;
        else
            return (temp*temp)/x;
    }
}

double spf_abs(double x)
{
    if(x < 0)
    {
        return -x;
    }

    return x;
}

double powerf(double base, double exponent)
{
    double result;
    int sign = 1;
    double x = 0;
    if (base < 0)
    {
        sign = -1;
        base = fabs(base);
    }

    if (exponent == 0.0)
    {
         return (1.0*sign);              // n**0 = 1
    }

    if ((base == 0.0) && (exponent > 0.0))
    {
         return (0.0);                   // 0**n = 0, n > 0
    }

    x = exponent - trunc (exponent);

    if ((x == 0.0) && (fabs(exponent) <= 2147483647))
    {
        result = (sign*poweri(base, (int)trunc(exponent)));
    }
    else
    {
        result = (sign*exp(exponent * log(base)));
    }
    return result;
}

double logd(double base, double number)
{
    return std::log(number) / std::log(base);
}

double factoriald(double a)
{

    if (a <= 0 )
    {
        return 1;
    }
    return a * factoriald(a - 1);
}

int factoriali(int a)
{

    if (a <= 0 )
    {
        return 1;
    }
    return a * factoriali(a - 1);
}

double rootd(double a, double b)
{
    if(a != 0)
    {
        return std::pow(b, 1.0/a);
    }
    else
    {
        return 1.0;
    }
}


double sec(double a)
{
    return 1./std::cos(a);
}

//// Cotangent
double cot(double a)
{
    return 1./std::tan(a);
}
//
//// Inverse cotangent
double arccot(double a)
{
    double result;
    if(a == 0.0)
    {
        result = M_PI/2.;
    }
    else
    {
        result = atan(1. / a);
    }
    return result;
}

double arccot_negzero(double a)
{
    double result;
    if(a == 0.0)
    {
        result = -M_PI/2.;
    }
    else
    {
        result = atan(1. / a);
    }
    return result;
}

//// Inverse cotangent - ratio numerator and denominator provided
double arccot2(double a, double b)
{
    return 0;
}

//// Inverse secant
double arcsec(double a)
{
    return std::acos(1.0 / a);
}

//// Cosecant
double csc(double a)
{
    return 1.0 / std::sin(a);
}

//// Inverse cosecant
double arccsc(double a)
{
     return std::asin(1.0 / a);
}

//// Hyperbolic secant of a double number
double sech(double a)
{
    return 1.0 / std::cosh(a);
}

//// Inverse hyperbolic secant of a double number
double asech(double a)
{
    return std::log(1.0 / a + std::sqrt(std::pow(1.0 / a, 2.0) + 1.0));
}

double arcsech(double z)
{
    return std::log(1.0/z + std::sqrt(1.0/z + 1)*std::sqrt(1.0/z - 1));
}

//// Hyperbolic cosecant of a double number
double csch(double a)
{
    return 1.0 / std::sinh(a);
}

//// Inverse hyperbolic cosecant of a double number
double arccsch(double z)
{
    return std::log(1.0/z + std::sqrt(1.0/(z*z) + 1));
}

//// Hyperbolic cotangent of a double number
double coth(double a)
{
    return std::cosh(a) / std::sinh(a);
}

//// Inverse hyperbolic cotangent of a double number
double arccoth(double a)
{
    return (std::log(1.0 + (1.0 / a)) - std::log(1.0 - (1.0 / a))) / 2.0;
}

//// Inverse hyperbolic functions
//// --------------------------------------------------------------
//// Inverse hyperbolic sine of a double number
double arcsinh(double a)
{
    return std::log(a + std::sqrt(std::pow(a, 2.0) + 1.0));
}

//// Inverse hyperbolic cosine of a double number
double arccosh(double a)
{
    return std::log(a + (std::sqrt(std::pow(a, 2.0) - 1.0)));
}

//// Inverse hyperbolic tangent of a double number
double arctanh(double a)
{
    return (std::log(1.0 + a) - std::log(1.0 - a)) / 2.0;
}

}
} /* namespace sbmlsupport */
