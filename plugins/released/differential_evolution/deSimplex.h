//       D I F F E R E N T I A L     E V O L U T I O N                                                                
//Original Authors: Dr. Rainer Storn and Kenneth Price.
//Modified by: Debashish Roy.
//Rainer Storn and Ken Price as the originators of the the DE idea.   
#pragma once

#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <random>
#include <iomanip>
#include <utility>
#include <memory>
#include <limits>

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "memory.h"
#include "deDifferentialEvolution.h"

#define MAXPOP  500
#define MAXDIM  35

using namespace std;

double c[MAXPOP][MAXDIM], d[MAXPOP][MAXDIM];
double oldarray[MAXPOP][MAXDIM];
double newarray[MAXPOP][MAXDIM];
double swaparray[MAXPOP][MAXDIM];

int CopyVector(double a[], double b[]) {
  for (int k=0; k<MAXDIM; k++) {
    a[k] = b[k];
  }
  return 0;
}

int CopyArray(double dest[MAXPOP][MAXDIM], double src[MAXPOP][MAXDIM]) {
  for (int j=0; j<MAXPOP; j++) {
    for (int k=0; k<MAXDIM; k++) {
      dest[j][k] = src[j][k];
    }
  }
  return 0;
}

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);
double uniform()	//uniform real distribution btw 0 and 1
{
	return dis(gen);
}

double simplex2(
double (*evaluate)(double[], const void* userData),
const void* userData,       //&hmost
double start[],         //initial parameters
int D,                  //no of parameters
double CR,              //cross over propability
double F,               //Differential weight
void (*constrain)(double[],int n),//null
int maxIterations,
double inibound_h=10000,
double inibound_l=-10000)
{
	inibound_h= std::numeric_limits<double>::infinity();
	inibound_l= (-1)*std::numeric_limits<double>::infinity();
    int NP=10*D;
    int genmax=500;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	//std::default_random_engine gen;
    std::uniform_real_distribution<double> dis(0.0, 1.0);//dis(gen) will genereated uniform real distribution
    double energy[MAXPOP];  // obj. funct. values of ith candidate sol  
    double tmp[MAXDIM], best[MAXDIM], bestit[MAXDIM]; // members 
    double r;
	//cout << dis(gen);
    int   i, j, L, n;      // counting variables                 
    int   r1, r2, r3;  // placeholders for random indexes       
    int   imin;            // index to member with lowest energy     
    double trial_energy;    // buffer variable                 
    double emin,random_no;            // help variables                

    // spread initial population members
    for(j=0;j<D;++j){
        c[0][j]=start[j];
    }
    energy[0]=evaluate(c[0],userData);
    for(i=1; i<NP; i++){
        for(j=0; j<D; j++){
			r =uniform();
            c[i][j]=inibound_l+r*(inibound_h-inibound_l);
        }
        energy[i]=evaluate(c[i],userData);
    }
	
    emin=energy[0];
    imin=0;
    for(i=1;i<NP;i++){
        if(energy[i]<emin){
            emin = energy[i];
            imin = i;
        }
    }

    CopyVector(best, c[imin]);
    CopyVector(bestit, c[imin]);

    // old population (generation G)
    // new population (generation G+1)
    CopyArray(oldarray, c);
    // new population (generation G+1)
    CopyArray(newarray, d);

    // Iteration loop to optimize
    for(int gen=0;gen<maxIterations;++gen)
    {
        imin=0;

        for(i=0;i<NP;i++){
            // Pick a random population member 
            do{
				random_no = uniform();
				random_no *= NP;
				r1 = random_no;
			} while (r1 == i);
            do{
				//r2 = (int)(dis(gen)*NP); not working
				random_no = uniform();
				random_no *= NP;
				r2 = random_no;
			} while (r1 == r2 || i == r2);
            do{
				//r3 =(int)(dis(gen)*NP);
				random_no = uniform();
				random_no *= NP;
				r3 = random_no;
			} while (r1 == r3 || i == r3 || r2 == r3);

            for(int k=0;k<MAXDIM;k++){
                tmp[k]=oldarray[i][k];
            }
			//n = (int)(dis(gen)*D);
			random_no = uniform();
			random_no *= D;
			r1 = random_no;
            L=0;
            do{                       //mutation
                tmp[n] = oldarray[r1][n] + F*(oldarray[r2][n] - oldarray[r3][n]);
                n = (n+1)%D;
                L++;
			} while ((uniform() < CR) && (L < D));

            // Trial mutation now in tmp[]. Test how good this choice really was.
            // Evaluate new vector in tmp[]
            trial_energy=evaluate(tmp,userData);  
            // improved objective function value?
            if (trial_energy <= energy[i]) {        //crossover                           
                energy[i]=trial_energy;         
                for(int k=0; k<MAXDIM; k++){
                    newarray[i][k]=tmp[k];
                }
                // Was this a new minimum?
                if (trial_energy<emin){
                // reset emin to new low...
                    emin=trial_energy;           
                    imin=i;
                    for(int k=0; k<MAXDIM; k++){
                        best[k] = tmp[k];
                    }         
                }                           
            } 
            else {
                // replace target with old value
                for (int k=0; k<MAXDIM; k++) {
                    newarray[i][k]=oldarray[i][k];
                }
            }
        }

        CopyVector(bestit, best);  // Save best population member of current iteration

        // swap population arrays. New generation becomes old one
        CopyArray(swaparray, oldarray);
        CopyArray(oldarray, newarray);
        CopyArray(newarray, swaparray);

        /*if (false) {
            printf("\n\n Best-so-far obj. funct. value: %15.10f",emin);
            for (j=0; j<D; j++) {
                printf("\n best[%d]: %14.7f", j, best[j]);
            }
            printf("\n Generation: %d  NFEs: %d", gen, nfeval);
            printf("\n Strategy: %d  NP: %d  F: %f  CR: %f", strategy, NP, F, CR);
        }*/
    }
    return emin;
}
