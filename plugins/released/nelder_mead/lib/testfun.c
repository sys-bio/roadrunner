/* 
 * Program: testfun.c
 * Author: Michael F. Hutt
 * http://www.mikehutt.com
 * Mar. 1, 2011
 * gcc -Wall -o testfun testfun.c nmsimplex.c -DMACOSX
 * test function for nmsimplex
 *
 */
 
 #include "nmsimplex.h"
 #include <stdio.h>
 #include <math.h>
 
 double rosen(double x[])
 {
 	return (100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1.0-x[0])*(1.0-x[0]));
 }

 double round2(double num, int precision)
 {
 	double rnum;
 	int tnum;

 	rnum = num*pow(10,precision);
 	tnum = (int)(rnum < 0 ? rnum-0.5 : rnum + 0.5);
 	rnum = tnum/pow(10,precision);

 	return rnum;
 }

 void my_constraints(double x[], int n)
 {
   // round to 2 decimal places
   int i;

   for (i=0; i<n; i++) {
     x[i] = round2(x[i],2);
   }
 }
 
 int main()
 {
 	double start[] = {-1.2,1.0};
 	double min;
 	int i;

 	min=simplex(rosen,start,2,1.0e-4,1,my_constraints);
 	//min=simplex(rosen,start,2,1.0e-4,1,NULL);

 	for (i=0;i<2;i++) {
 		printf("%f\n",start[i]);
 	}
 	return 0;
 }