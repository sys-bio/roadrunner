#include <iostream>
#include "telException.h"
#include "telUtils.h"
#include "telProperty.h"
#include "telProperties.h"
#include "telException.h"
#pragma hdrstop


using namespace std;
using namespace tlp;

int main()
{
    try
    {
        //Create some properties
        Property<double> aProperty1(23.45, "TestProp1", "The Hint", "The Descr", "TheAlias1", true);
        Property<double> aProperty2(3.45,  "TestProp2", "The Hint", "The Descr", "TheAlias2", true);

        //Create a property container, and add the propeties to it
        Properties props1;
        props1.add(&aProperty1);
        props1.add(&aProperty2);

        cout<<props1;
        aProperty1.setValue(1023);
        cout<<props1;

        Properties props2;
        props2 = props1;
        cout<<props2;

    }
    catch(const Exception& ex)
    {
        cout<<ex.what();
    }
}
