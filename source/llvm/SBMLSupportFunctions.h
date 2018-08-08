/*
 * SBMLSupportFunctions.h
 *
 *  Created on: Aug 22, 2013
 *      Author: andy
 */

#ifndef SBMLSUPPORTFUNCTIONS_H_
#define SBMLSUPPORTFUNCTIONS_H_

namespace sbmlsupport
{
extern "C" {

double trunc(double d);

double poweri(double x, int y);

double spf_abs(double x);

double powerf(double base, double exponent);

int factoriali(int a);

double factoriald(double a);

double rootd(double a, double b);

double logd(double base, double number);

double sec(double a);

//// Cotangent
double cot(double a);
//
//// Inverse cotangent
double arccot(double a);

/**
 * flavor of arccot that returns -pi/2 for 0.
 *
 * Idiotic hack to pass test case 959
 */
double arccot_negzero(double a);

//// Inverse cotangent - ratio numerator and denominator provided
double arccot2(double a, double b);

//// Inverse secant
double arcsec(double a);

//// Cosecant
double csc(double a);

//// Inverse cosecant
double arccsc(double a);

//// Hyperbolic secant of a double number
double sech(double a);

//// Inverse hyperbolic secant of a double number
double arcsech(double a);

double arcsech(double a);

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

//// Quotient of division of two double numbers
double quotient(double a, double b);

//// Maximum of two double numbers
double max(double a, double b);

//// Minimum of two double numbers
double min(double a, double b);
}

} /* namespace sbmlsupport */
#endif /* SBMLSUPPORTFUNCTIONS_H_ */
