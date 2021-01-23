#include "nmsimplex.h"
#include <stdio.h>
#include <math.h>

#pragma comment(lib, "nelder_mead-static.lib")
double objfun(double x[])
{
    /* find the power for this value of RL */
    double Req;
    double VL;
    double PL;
    double Vs = 9.0;
    double RL = x[0];

    Req = 470*RL/(470+RL);
    VL = Vs*Req/(1000+Req);
    PL = VL*VL/RL;
    printf("RL:%f P:%f\n",RL,PL);
    return -PL; /* nm finds the minimum, so to find the max we return the negative value */
}

void my_constraints(double x[], int n)
{
    // resistance must be positive
    int i;
    for (i=0; i<n; i++)
    {
        if (x[i] < 0)
        {
            x[i] = fabs(x[i]);
        }
    }
}

int main()
{
    double start[] = {100};
    double min;
    int i;
    int dim = 1;
    double eps = 1.0e-8;
    double scale = 1.0;

    min=simplex(objfun,start,dim,eps,scale,my_constraints);

    for (i=0; i<dim; i++)
    {
        printf("%f\n",start[i]);
    }
    return 0;
}

/*
 * Program: maxpower.c
 * Author: Michael F. Hutt
 * http://www.mikehutt.com
 * Mar. 2, 2011
 * gcc -Wall -o maxpower maxpower.c nmsimplex.c -DMACOSX
 *
 * Example to test out nelder-mead on a one dimensional optimization problem
 *
 * The circuit used for this example is a voltage divider consisting of a 1k ohm
 * and 470 ohm resistor with a 9V source. The output is take across the 470 ohm resistor.
 *
 * spice file circuit description
 * Vs 1 0 9
 * R1 1 2 1000
 * R2 2 0 470
 * RL 2 0 320
 *
 * The Thevenin equivalent resistance is 319.73 ohms, which is the value we should find
 * from the optimization.
 *
 *
 * RL:320.000000 P:0.006474
 * 29 Function Evaluations
 * 13 Iterations through program
 *
 */

//OUTPUT
//RL:100.000000 P:0.004700
//RL:101.000000 P:0.004725
//Initial Values
//100.000000 -0.004700
//101.000000 -0.004725
//RL:102.000000 P:0.004749
//RL:103.000000 P:0.004773
//Iteration 1
//103.000000 -0.004773
//101.000000 -0.004725
//RL:105.000000 P:0.004820
//RL:107.000000 P:0.004866
//Iteration 2
//103.000000 -0.004773
//107.000000 -0.004866
//RL:111.000000 P:0.004954
//RL:115.000000 P:0.005039
//Iteration 3
//115.000000 -0.005039
//107.000000 -0.004866
//RL:123.000000 P:0.005196
//RL:131.000000 P:0.005339
//Iteration 4
//115.000000 -0.005039
//131.000000 -0.005339
//RL:147.000000 P:0.005588
//RL:163.000000 P:0.005792
//Iteration 5
//163.000000 -0.005792
//131.000000 -0.005339
//RL:195.000000 P:0.006094
//RL:227.000000 P:0.006288
//Iteration 6
//163.000000 -0.005792
//227.000000 -0.006288
//RL:291.000000 P:0.006460
//RL:355.000000 P:0.006457
//Iteration 7
//291.000000 -0.006460
//227.000000 -0.006288
//RL:355.000000 P:0.006457
//RL:323.000000 P:0.006474
//Iteration 8
//291.000000 -0.006460
//323.000000 -0.006474
//RL:355.000000 P:0.006457
//RL:307.000000 P:0.006472
//Iteration 9
//307.000000 -0.006472
//323.000000 -0.006474
//RL:339.000000 P:0.006469
//RL:315.000000 P:0.006474
//Iteration 10
//315.000000 -0.006474
//323.000000 -0.006474
//RL:331.000000 P:0.006473
//RL:319.000000 P:0.006474
//Iteration 11
//319.000000 -0.006474
//323.000000 -0.006474
//RL:315.000000 P:0.006474
//RL:321.000000 P:0.006474
//Iteration 12
//319.000000 -0.006474
//321.000000 -0.006474
//RL:317.000000 P:0.006474
//RL:320.000000 P:0.006474
//Iteration 13
//319.000000 -0.006474
//320.000000 -0.006474
//The minimum was found at
//3.200000e+02
//RL:320.000000 P:0.006474
//29 Function Evaluations
//13 Iterations through program
//320.000000
//



























