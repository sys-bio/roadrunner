#ifndef lmObjectiveFunctionH
#define lmObjectiveFunctionH
//---------------------------------------------------------------------------

void lmObjectiveFunction(
const double *par,       //Parameter vector
int           m_dat,     //Dimension of residue vector
const void   *userData,        //User data structure
double       *fvec,      //residue vector..
int          *userBreak  //Non zero value means termination
);

#endif
