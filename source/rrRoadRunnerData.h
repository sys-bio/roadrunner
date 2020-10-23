#ifndef rrRoadRunnerDataH
#define rrRoadRunnerDataH

#ifdef _MSC_VER
#pragma warning(disable: 26812)
#pragma warning(disable: 26451)
#endif
#include "rr-libstruct/lsMatrix.h"
#ifdef _MSC_VER
#pragma warning(disable: 26812)
#pragma warning(disable: 26451)
#endif

#include "rrExporter.h"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace rr
{

using namespace ls;
using std::ofstream;
using std::stringstream;

class RoadRunner;

/**
 * @deprecated
 *
 * DO NOT USE THIS, IT IS DEPRECATED AND WILL BE REMOVED.
 */
class RR_DECLSPEC RoadRunnerData
{

public:
    RoadRunnerData(const int& rSize = 0, const int& cSize = 0);

    RoadRunnerData(const std::vector<std::string>& colNames,
            const DoubleMatrix& data);

    RoadRunnerData(const RoadRunner* rr);

    ~RoadRunnerData();

    void allocate(const size_t& cSize, const size_t& rSize);

    void allocateWeights();

    bool hasWeights() const;

    const std::vector<std::string>& getColumnNames() const;
    std::string getColumnName(const int col) const;
    std::string getColumnNamesAsString() const;
    void setColumnNames(const std::vector<std::string>& colNames);
    std::ptrdiff_t getColumnIndex(const std::string& colName) const;
    void setTimeDataPrecision(const int& prec);
    void setDataPrecision(const int& prec);
    void reSize(int rows, int cols);

    void clear();

    int rSize() const;

    int cSize() const;

    void setData(const DoubleMatrix& theData);

    bool loadSimpleFormat(const std::string& fileName);

    bool writeTo(const std::string& fileName) const;

    bool readFrom(const std::string& fileName);

    bool check() const;

    bool structuredResult;

    RR_DECLSPEC
    friend std::ostream& operator <<(std::ostream& ss, const RoadRunnerData& data);

    RR_DECLSPEC
    friend std::istream& operator >>(std::istream& ss, RoadRunnerData& data);

    double& operator()(const unsigned& row, const unsigned& col);

    double operator()(const unsigned& row, const unsigned& col) const;

    double getDataElement(int row, int col);

    void   setDataElement(int row, int col, double value);

    RoadRunnerData& operator=(const RoadRunnerData& rhs);

    double getWeight(int row, int col) const;

    void setWeight(int row, int col, double val);

    void setName(const std::string& name);

    std::string getName() const;

    std::pair<int, int> dimension() const;

    bool append(const RoadRunnerData& data);

    double getTimeStart() const;

    double getTimeEnd() const;

    const DoubleMatrix& getData() const;

    const DoubleMatrix& getWeights() const;


protected:

    /**
    * \brief Container holding column names.
    */
    std::vector<std::string> mColumnNames;

    /**
    * \brief Container holding the actual data.
    */
    DoubleMatrix mTheData;

    /**
    * \brief Container holding the data weights.
    */
    DoubleMatrix mWeights;

    /**
    * \brief Integer setting the precision of 'time' double numbers when writing to file
    */
    int mTimePrecision;

    /**
    * \brief Integer setting the precision of 'data' double numbers when writing to file
    */
    int mDataPrecision;            //The precision when saved to file

    /**
    * \brief String holding the 'name' of the object.
    * \todo Remove
    */
    std::string mName;             //For debugging purposes mainly..
};

}

#endif
