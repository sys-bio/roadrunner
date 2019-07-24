#ifndef telRandomH
#define telRandomH
#include "third_party/mtrand.h"
#include "telMathExporter.h"
//---------------------------------------------------------------------------

namespace rr
{

class MATH_DECLSPEC Random
{
    public:
                        Random(unsigned long seed);
        double          next() const;

    private:
        MTRand          mRand;        //Get a double in [0, 1)
};

}
#endif
