#pragma hdrstop
#include "tel_macros.h"
#include "rrplugins/common/telOSSpecifics.h"
#include "telplugins_telluriumdata_api.h"

#include "telplugins_utilities.h"
#include "telplugins_cpp_support.h"
#include "rrplugins/common/telStringUtils.h"
#include "rrplugins/common/telTelluriumData.h"
#include "rrplugins/common/telLogger.h"
#include "telplugins_cpp_support.h"


using namespace std;
using tlp::TelluriumData;
using tlp::StringList;

/*
#define start_try                                           \
        try                                                 \
        {
*/
/*!
 \brief macro for catch statement in a routine returning a bool
 \ingroup macros
*/
            /*
#define  tel_catch_bool_macro                                   \
    }                                                       \
    catch(exception& ex)                                    \
    {                                                       \
        stringstream msg;                                   \
        msg<<"Tellurium exception: "<<ex.what();            \
        tpSetError(msg.str());                                \
        return false;                                       \
    }*/

/*!
 \brief macro for catch statement in a routine returning a pointer
 \ingroup macros
*/
/*#define  tel_catch_ptr_macro                                    \
    }                                                       \
    catch(exception& ex)                                    \
    {                                                       \
        stringstream msg;                                   \
        msg<<"Tellurium exception: "<<ex.what();            \
        tpSetError(msg.str());                                \
        return NULL;                                        \
    }*/

/*!
 \brief macro for catch statement in a routine returning a positive integer
 \ingroup macros
*/
/*#define tel_catch_int_macro                                     \
    }                                                       \
    catch(exception& ex)                                    \
    {                                                       \
        stringstream msg;                                   \
        msg<<"Tellurium exception: "<<ex.what();            \
        tpSetError(msg.str());                                \
        return -1;                                          \
    }*/

/*!
 \brief macro for catch statement in a routine returning an unsigned integer
 \ingroup macros
*/
    /*
#define tel_catch_uint_macro                                     \
    }                                                       \
    catch(exception& ex)                                    \
    {                                                       \
        stringstream msg;                                   \
        msg<<"Tellurium exception: "<<ex.what();            \
        tpSetError(msg.str());                                \
        return 0;                                          \
    }*/

/*!
 \brief macro for catch statement in a routine returning void
 \ingroup macros
*/

/*
#define tel_catch_void_macro                                    \
    }                                                       \
    catch(const exception& ex)                              \
    {                                                       \
        stringstream msg;                                   \
        msg<<"Tellurium exception: "<<ex.what();            \
        tpSetError(msg.str());                                \
    }
*/


bool tlp_cc tpGetTelluriumDataElement(TELHandle handle, int row, int col, double* value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FILE__);
        *value = rrData->getDataElement(row, col);
        return true;
    tel_catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataElement(TELHandle handle, int row, int col, double value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FUNC__);
        rrData->setDataElement(row, col, value);
        return true;
    tel_catch_bool_macro
}

bool tlp_cc tpGetTelluriumDataWeight(TELHandle handle, int row, int col, double* value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FUNC__);

        if(rrData->hasWeights())
        {
            (*value) = rrData->getWeight(row, col);
            return true;
        }
        else
        {
            return false;
        }
    tel_catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataWeight(TELHandle handle, int row, int col, double value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FUNC__);
        if(rrData->hasWeights())
        {
            rrData->setWeight(row, col, value);
            return true;
        }
        else
        {
            return false;
        }
    tel_catch_bool_macro
}

bool tlp_cc tpHasWeights(TELHandle handle, bool* value)
{
    start_try
        TelluriumData* Data = castHandle< TelluriumData >(handle, __FUNC__);
        (*value) = Data->hasWeights();
        return true;
    tel_catch_bool_macro
}

bool tlp_cc tpAllocateWeights(TELHandle handle, bool* success)
{
    start_try
        TelluriumData* Data = castHandle< TelluriumData > (handle, __FUNC__);
        Data->allocateWeights();
        (*success) = Data->hasWeights();
        return true;
    tel_catch_bool_macro
}

char* tlp_cc tpGetTelluriumDataColumnHeader(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        char* text = tpCreateText(data->getColumnNamesAsString());
        return text;
    tel_catch_ptr_macro
}

char* tlp_cc tpGetTelluriumDataColumnHeaderByIndex(TELHandle handle, int index)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        StringList cols = data->getColumnNames();
        return tpCreateText(cols[index]);
    tel_catch_ptr_macro
}

bool tlp_cc tpSetTelluriumDataColumnHeader(TELHandle handle, char* hdr)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        StringList hdrList(hdr, ", ");
        return data->setColumnNames(hdrList);
    tel_catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataColumnHeaderByIndex(TELHandle handle, int index, char* hdr)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->setColumnName(index, hdr);
    tel_catch_bool_macro
}

int tlp_cc tpGetTelluriumDataNumRows(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->rSize();
    tel_catch_int_macro
}

int tlp_cc tpGetTelluriumDataNumCols(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->cSize();
    tel_catch_int_macro
}

TELHandle tlp_cc tpCreateTelluriumData(int nRows, int nCols, char* colNames)
{
    start_try
        TelluriumData* data = new TelluriumData(nRows, nCols);
        gHM.registerHandle(data, typeid(data).name());
        if (colNames)
        {
            string cNames(colNames);
            StringList colNames(cNames, ",");
            data->setColumnNames(colNames);
        }
        return data;
    tel_catch_ptr_macro
}

bool tlp_cc tpFreeTelluriumData(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        delete data;
        return true;
    tel_catch_bool_macro
}

bool tlp_cc tpWriteTelluriumDataToFile(TELHandle handle, const char* fName)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->write(fName);
    tel_catch_bool_macro
}

bool tlp_cc tpReadTelluriumDataFromFile(TELHandle handle, const char* fName)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        //Check if file exists first
        return data->read(fName);
    tel_catch_bool_macro

}
