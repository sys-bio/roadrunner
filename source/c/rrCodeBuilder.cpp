#pragma hdrstop
#include <iomanip>
#include "rrStringUtils.h"
#include "rrCodeBuilder.h"
//---------------------------------------------------------------------------

namespace rr
{

CodeBuilder::CodeBuilder(const std::string& aStr, const std::string& decl_spec, const std::string& call_conv)
:
mSizeOfVarField1(45),
mSizeOfVarField2(55),
mSizeOfVarField3(45),
mDeclSpec(decl_spec),
mCallingConvention(call_conv)
{
    mStringing<<aStr;
}

void CodeBuilder::FormatVariable(const std::string& type, const std::string& varName, const std::string& comment)
{

    mStringing<<std::left<<std::setw(mSizeOfVarField1)<<type    <<varName<<     std::setw(mSizeOfVarField2)<<";";
    if(comment.size())
    {
        mStringing<<"//"<<comment;
    }

    mStringing<<std::endl;
}

void CodeBuilder::AddFunctionExport(const std::string& retValue, const std::string& funcProto)
{
    //mStringing<<mDeclSpec<<" "<<std::left<<std::setw(mSizeOfVarField1)<<retValue<<mCallingConvention<<std::setw(mSizeOfVarField2)<<funcProto + ";"<<std::endl;
    mStringing<<mDeclSpec<<" "<<std::left<<std::setw(mSizeOfVarField1)<<retValue<<std::setw(mSizeOfVarField2)<<funcProto + ";"<<std::endl;
}

void CodeBuilder::AddFunctionProto(const std::string& retValue, const std::string& funcProto)
{
    mStringing<<"   "<<" "<<std::left<<std::setw(mSizeOfVarField1)<<retValue<<std::setw(mSizeOfVarField2)<<funcProto + ";"<<std::endl;
}

void CodeBuilder::FormatArray(const std::string& type, const std::string& varName, const int& _arraySize, const std::string& comment)
{
    int arraySize = _arraySize;
    if(arraySize == 0)
    {
        //an array of zero length is undefined.. don't put it in the header..
        mStringing<<"\n//The array size for the follwoing variable was generated as 0. We put 1, to make it legal code.\n";
         arraySize = 1;
    }

    std::string field2(varName +"["+ rr::toString(arraySize)+"];");
    mStringing<<std::left<<std::setw(mSizeOfVarField1)<<type    << std::setw(mSizeOfVarField2)<<field2;

    if(comment.size())
    {
        mStringing<<std::left<<std::setw(mSizeOfVarField3)<<"//" + comment;
    }
    mStringing<<"\n";

    //Add the size for each array, so we don't have to calculate later on..
    if(_arraySize == 0)
    {
        arraySize = 0;
    }

    mStringing<<std::left<<std::setw(mSizeOfVarField1)<<"D_S const int"    << std::setw(mSizeOfVarField2)<<varName + "Size=" + rr::toString(arraySize) + ";";
    mStringing<<std::endl;
}

}
