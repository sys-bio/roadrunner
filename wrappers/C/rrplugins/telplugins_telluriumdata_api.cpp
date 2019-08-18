#pragma hdrstop
#include "rrRoadRunner.h"
#include "rrOSSpecifics.h"
#include "telplugins_telluriumdata_api.h"
#include "tel_macros.h"
#include "telplugins_utilities.h"
#include "telplugins_cpp_support.h"
#include "rrStringUtils.h"
#include "rrplugins/common/telTelluriumData.h"
#include "rrLogger.h"

using rr::RoadRunner;
using namespace std;
using rr::TelluriumData;
using rr::StringList;

bool tlp_cc tpGetTelluriumDataElement(TELHandle handle, int row, int col, double* value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FILE__);
        *value = rrData->getDataElement(row, col);
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataElement(TELHandle handle, int row, int col, double value)
{
    start_try
        TelluriumData* rrData = castHandle< TelluriumData >(handle, __FUNC__);
        rrData->setDataElement(row, col, value);
        return true;
    catch_bool_macro
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
    catch_bool_macro
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
    catch_bool_macro
}

bool tlp_cc tpHasWeights(TELHandle handle, bool* value)
{
    start_try
        TelluriumData* Data = castHandle< TelluriumData >(handle, __FUNC__);
        (*value) = Data->hasWeights();
        return true;
    catch_bool_macro
}

bool tlp_cc tpAllocateWeights(TELHandle handle, bool* success)
{
    start_try
        TelluriumData* Data = castHandle< TelluriumData > (handle, __FUNC__);
        Data->allocateWeights();
        (*success) = Data->hasWeights();
        return true;
    catch_bool_macro
}

char* tlp_cc tpGetTelluriumDataColumnHeader(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        char* text = tpCreateText(data->getColumnNamesAsString());
        return text;
    catch_ptr_macro
}

char* tlp_cc tpGetTelluriumDataColumnHeaderByIndex(TELHandle handle, int index)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        StringList cols = data->getColumnNames();
        return tpCreateText(cols[index]);
    catch_ptr_macro
}

bool tlp_cc tpSetTelluriumDataColumnHeader(TELHandle handle, char* hdr)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        StringList hdrList(hdr, ", ");
        return data->setColumnNames(hdrList);
    catch_bool_macro
}

bool tlp_cc tpSetTelluriumDataColumnHeaderByIndex(TELHandle handle, int index, char* hdr)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->setColumnName(index, hdr);
    catch_bool_macro
}

int tlp_cc tpGetTelluriumDataNumRows(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->rSize();
    catch_int_macro
}

int tlp_cc tpGetTelluriumDataNumCols(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->cSize();
    catch_int_macro
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
    catch_ptr_macro
}

bool tlp_cc tpFreeTelluriumData(TELHandle handle)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        delete data;
        return true;
    catch_bool_macro
}

bool tlp_cc tpWriteTelluriumDataToFile(TELHandle handle, const char* fName)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        return data->write(fName);
    catch_bool_macro
}

bool tlp_cc tpReadTelluriumDataFromFile(TELHandle handle, const char* fName)
{
    start_try
        TelluriumData* data = castHandle< TelluriumData >(handle, __FUNC__);
        //Check if file exists first
        return data->read(fName);
    catch_bool_macro

}
