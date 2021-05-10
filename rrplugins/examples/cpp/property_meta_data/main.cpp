#include <iostream>
#include "telException.h"
#include "telUtils.h"
#include "telProperty.h"
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

        //Check output
        cout<<aProperty1<<"\n\n"<<aProperty2;

        //Create a property container, and add the propeties to it
        Properties props;
        props.add(&aProperty1);
        props.add(&aProperty2);


        //Retrieve a property using an alias.
        PropertyBase* aProp = props.getProperty("TheAlias2");

        if(!aProp)
        {
            throw(Exception("Could not find property"));
        }

        //Check we got it right
        cout<< "\n\n" << (*aProp);
        cout<<"Is read only: "<<toString(aProp->isReadOnly());
    }
    catch(const Exception& ex)
    {
        cout<<ex.what();
    }
}
