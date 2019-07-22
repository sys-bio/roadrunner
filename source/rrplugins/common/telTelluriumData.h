#ifndef telTelluriumDataH
#define telTelluriumDataH
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "../../c/rrStringList.h"
#include "telCommonExporter.h"
#include "telArrayedParameter.h"
//#include "telMatrix.h"

namespace rr
{
using namespace ls;
using std::ofstream;
using std::stringstream;

/**
 * \brief TelluriumData is a general purpose container for numerical data, e.g. simulation output.
 \section telData TelluriumData
 * The class provide the ability to read and write data from file (see Format specification below).
 * For certain problems, weights can be specified as a property of the object as well. Weights are not allocated automatically, but can
 * be allocated using the allocateWeights() member function.
 * In addition to the data (internally stored as a matrix), a column header property is available.
 * The column header store information on the content in data columns, e.g time, S1, S2, etc..

 * \subsection dataFormat DataFormat
 * When TelluriumData is written or read from file, a pre defined data format is used. This format
 * is defined here.
 *
 * The following shows the format of a TelluriumData object as written to file
    \code
        [INFO]
        DATA_FORMAT_VERSION=1.0
        CREATOR=libRoadRunner
        NUMBER_OF_COLS=3
        NUMBER_OF_ROWS=5
        COLUMN_HEADERS=time,[S1],[S2]
        COMMENT="Model xxx was used to generate this data"

        [DATA]
        0,0.0001593016905797244,-7.559109749054044e-006
        0.0195695,0.000147779593688827,-2.827795051612286e-007
        0.0391389,0.0001538108562389516,-4.24906368628629e-006
        0.0587084,0.0001284775775414624,3.978631908960495e-006
        0.0782779,0.0001347344766870294,7.306180621780478e-006
    \endcode

    As seen above, the file is divided into sections, enclosed in square brackets, i.e. [INFO] and [DATA].

    The [INFO] section contains general information. Each line is described below
    \li DATA_FORMAT_VERSION=1.0  Contains a version number of the format used writing the file. Currently unused.
    \li CREATOR=libRoadRunner  Contain a description of who, or what created the data file. Currently unused.
    \li NUMBER_OF_COLS=3  Contain the number of columns in the data.
    \li NUMBER_OF_ROWS=5  Contain the number of rows in the data.
    \li COLUMN_HEADERS=time,[S1],[S2]  Contain descriptions of each data column.

    The [DATA] section hold the actual numerical data. It is written row by row as comma seperated values.

    \subsubsection weights Weights
    The data file can optionally store data weights. Weights are numbers used to 'weigh' individual data numbers, which is
    useful in various problems.
    In the case of using weighted data, an additional section named [WEIGHTS] would be found in the data file, i.e.

    \code
        [INFO]
        DATA_FORMAT_VERSION=1.0
        .
        .

        [DATA]
        0,0.0001593016905797244,-7.559109749054044e-006
        .
        .

        [WEIGHTS]
        1,1,1
        1,1,0.34
        1,1.23,1
        1,1,0.45
        1,1,1
    \endcode

    The [WEIGHTS] section is required to have the same dimension as the [DATA] section.
    Weigts are read into the internal data container, named mWeights, and accessed trough the
    member functions getWeights(), weight(row, col) and setWeight(row,col).

    As an example of how weights can be used is shown below:

   \code
   ....
    if(myData.hasWeights())
    {
        for (row...
         for (col...
        myStatisticsData(row,col) = myData.weight(row, col) * myData(row, col);
    }
   ....
   \endcode

 *
 *
 * \note If the size of the column header differ from the column size of the data, the writeTo function will fail and return false.
 * For a client to successfully write data to file, the size of the column header is required to match the actual size of data columns.
 *
 * If data is read from file, and the size of the COLUMNM_HEADERS don't match the underlying data, reading will fail
 * and an exception will be thrown.
 *
 */

class COMMON_DECLSPEC TelluriumData
{
    public:
        /**
         * \brief Constructor Create a Tellurium data object with dimension rSize x cSize.
         *
         * \param rSize: number of rows
         * \param cSize: number of columns.
         */
        TelluriumData(const int& rSize = 0, const int& cSize = 0);

        /**
         * \brief Constructor -- copy the names and data from the given
         * parameters.
         *
         * \param colNames List of column names
         * \param data Source data matrix.
         */
        TelluriumData(const StringList& colNames, const rr::DoubleMatrix& data);

        /**
         * \brief Copy Constructor Create a Tellurium data object from a Tellurium data object
         *
         * \param data: Pointer to a Telluriume data object
         */
        TelluriumData(const TelluriumData* data);

        /**
         * \brief Copy Constructor Create a Tellurium data object from a Tellurium data object
         *
         * \param data: A Telluriume data object
         */
        TelluriumData(const TelluriumData& data);

        /**
         *   \brief Destructor. De allocate any memory allocated in the class.
         */
        ~TelluriumData();

        /**
         *   \brief allocate. Allocate an underlying data matrix of size cSize x rSize.
         */
        void allocate(const int& cSize, const int& rSize);

        /**
         *   \brief allocate. Allocate an underlying data (weights) matrix of size cSize x rSize.
         */
        void allocateWeights();

        /**
         *   \brief hasWeights returns true or false indiciating if current object have weights allocated or not.
         */
        bool hasWeights() const;

        /**
         * \brief Get the names of the variables stored in this data structure, as a StringList
         */
        const StringList&                       getColumnNames() const;

        bool                                    isFirstColumnTime() const;
        std::string                             getColumnName(const int col) const;
        std::string                             getColumnNamesAsString() const;
        bool                                    setColumnNames(const StringList& colNames);
        bool                                    setColumnName(int index, const string& colName);
        int                                     getColumnIndex(const std::string& colName) const;
        void                                    setTimeDataPrecision(const int& prec);
        void                                    setDataPrecision(const int& prec);
        void                                    reSize(int rows, int cols);
        string                                  getComments() const;
        void                                    setComments(const string& coms);
        void                                    swapRows(int r1, int r2);
        /**
         * Clear all data allocated
         */
        void clear();

        /**
         * \brief Return the number of rows in the underlying data matrix.
         */
        int rSize() const;

        /**
         * \brief Return the number of columns in the underlying data matrix
         */
        int cSize() const;

        /**
         * \brief Assign data from an external DoubleMatrix.
         */
        void setData(const DoubleMatrix& theData);

        /**
         * \brief Read data from a csv text file. This function can read
         * the data format used in the libSBML test suite for example.
         * Observe that both comma, ',' and space, ' ', or both are valid as value separators.
         * The input data format is as follows:
         *
         * First line, comma separated values defining the column header, e.g. "time, S1, S2" etc
         *
         * Second line to end of file: Numeric values in comma (and/or space) separated columns, corresponding to the
         * column header
         * \code
               time,S1,S2
               0,0.00015,0
               0.1,0.0001357256127053939,1.427438729460607e-005
               0.2,0.0001228096129616973,2.719038703830272e-005
               0.3,0.0001111227331022577,3.887726689774233e-005
               .......
           \endcode
         *
         *
         */
        bool readCSV(const std::string& fileName);

        /**
            \brief Write data to file. See required format in the TelluriumData general description.
            \param fileName Name, including path, for the output file.
            \todo Rename to writeToFile
        */
        bool write(const std::string& fileName) const;

        /**
            \brief Read data from file. See required format in the TelluriumData general description.
            \param fileName Name, including path, for the output file.
            \todo Rename to readFromFile
        */
        bool read(const std::string& fileName);

        /**
         * \brief Validate data format. Basically checks if the objects column names container size
         * correspond to the size of the actual data.
         * \todo Rename to validateFormat()
         */
        bool check() const;

        /**
        * \brief Stream friend function, allowing Tellurium data to be streamed to a std::ostream.
        */
        COMMON_DECLSPEC
        friend std::ostream& operator <<(std::ostream& ss, const TelluriumData& data);

        /**
        * \brief Stream friend function, allowing Tellurium data to be streamed from a std::istream.
        */
        COMMON_DECLSPEC
        friend std::istream& operator >>(std::istream& ss, TelluriumData& data);

        /**
        * \brief Access data element operator.
        *
        * The following code will assign 45 to data at row 3 and column 4
        \code
         myData(3,4) = 45;
        \endcode

        * \param row Index of row
        * \param col Index of col
        */

        double& operator()(const unsigned& row, const unsigned& col);

        /**
        * \brief Access data element operator used in constant expression.
        *
        * The following code will print the element at row 3 and column 4
        \code
         cout<< myData(3,4);
        \endcode

        * \param row Index of row
        * \param col Index of col
        */
        double operator()(const unsigned& row, const unsigned& col) const;

        /**
        * \brief Return data element at (row, col)

        * \param row Index of row
        * \param col Index of col
        */
        double getDataElement(int row, int col);

        /**
        * \brief Set data element at (row, col)

        * \param row Index of row
        * \param col Index of col
        * \param value New value
        */
        void   setDataElement(int row, int col, double value);

        /**
        * \brief Assignment operator. Deep copy of data in an assignment expression.
        *
        * \param rhs TelluriumData object on the right side of the assignment ().
        */
        TelluriumData& operator=(const TelluriumData& rhs);

        /**
        * \brief Access a weight element.
        *
        * The following code will print the weight element at row 3 and column 4
        \code
         cout<< myData.getWeight(3,4);
        \endcode
        * \param row Index of row
        * \param col Index of col
        */
        double getWeight(int row, int col) const;

        /**
        * \brief Set data weight at number at row,col index.
        * \param row Index of row
        * \param col Index of col
        * \param val Value of new weight
        */
        void setWeight(int row, int col, double val);

        /**
        * \brief Set data object name.
        * \param name The name as a string
        * \deprecated Left over function from debugging session ??
        */
        void setName(const std::string& name);

        /**
        * \brief Get data object name.
        * \deprecated Left over function from debugging session ??
        */
        std::string getName() const;

        /**
        * \brief Return the dimension of underlying data.
        */
        std::pair<int, int> dimension() const;

        /**
        * \brief Append data columns. In certain circumstances the client
        * modify some initial data, say S1, S2 etc. and obtain a new set of data, S1', S2' etc.
        * The append function creates a new data set, containing the data columns in the argument
        * data as appended columns, e.g. if the original data contain
        data columns S1, S2, and the data in the argument contain S1', S2', the resulting data
        set will have S1, S2, S1', S2' as the columns.

        *This function require the row size of the two data sets to be equal.
        */
        bool append(const TelluriumData& data);

        /**
        * \brief If the data has a column named 'time', the function will return the first time element.
        * If no time column exists, then a DoubleNaN is returned
        */
        double getTimeStart() const;

        /**
        * \brief If the data has a column named 'time', the function will return the last time element.
        * If no time column exists, then a DoubleNaN is returned
        */
        double getTimeEnd() const;

        /**
        * \brief Return a const reference to the underlying data matrix.
        *
        */
        const DoubleMatrix& getData() const;

        /**
        * \brief Return a const reference to the underlying data matrix hodling Weights.
        */
        const DoubleMatrix& getWeights() const;

        /**
            \brief The concept of arrayed parameters are under investigation..
        */
        ArrayedParameter    getArrayedParameter() const;
        void                setArrayedParameter(const ArrayedParameter& para);

        void byteCheck() const;

    protected:
        /**
        * \brief Container holding column names.
        */
        StringList mColumnNames;

        /**
        * \brief String containing comments about the data.
        */
        string mComments;

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
        * \brief Currently only one arrayed parameter is supported
        */
        ArrayedParameter    mArrayedParameter;

        unsigned int mBytePattern;

};

//When using tellurium data as a property the following template functions are needed
//============= Tellurium data ===========================
template<>
inline string Property<rr::TelluriumData>::getValueAsString() const
{
    std::stringstream rrData;
    rrData << (mValue);
    return rrData.str();
}

template<>
inline void Property< rr::TelluriumData >::setValue(const void* val)
{
    mValue = * ((rr::TelluriumData*) (val));
}

/**
    Set a RoadRunner data properties value, from a string.
    \note This is not implemented.
*/
template<>
inline void Property<rr::TelluriumData>::setValueFromString(const string& val)
{
}

template<>
inline bool Property<rr::TelluriumData>::clearValue()
{
    mValue = rr::TelluriumData();
    return true;
}

/**
    Set a property value
*/
template<>
inline void Property<rr::TelluriumData>::setValue(const rr::TelluriumData& val)
{
    mValue = val;
}

/**
    Set a property value
*/
template<>
inline void Property<rr::TelluriumData>::setValue(rr::TelluriumData* val)
{
    mValue = (*val);
}

/**
    \brief Returns the type as a string.
*/
template<>
inline string getPropertyType<TelluriumData>(const TelluriumData& a)
{
    return "telluriumData";
}

template<>
inline Property<TelluriumData>::operator TelluriumData () const
{
    return mValue;
}

template<>
inline Property<TelluriumData>::operator TelluriumData & ()
{
    return mValue;
}

}

#endif
