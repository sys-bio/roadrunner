#ifndef telRandomH
#define telRandomH
#include "third_party/mtrand.h"
#include "telMathExporter.h"
//---------------------------------------------------------------------------

namespace tlp
{

class MATH_DECLSPEC Random
{
    public:
                        Random(unsigned long seed);
        double          next() const;

                        //SEEDDIAG: state fingerprint of the shared generator
        static unsigned long long diagFingerprint();
        static const void*        diagStateAddress();

    private:
        MTRand          mRand;        //Get a double in [0, 1)
};

}
#endif
