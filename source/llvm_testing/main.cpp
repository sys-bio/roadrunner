//#include "tests.h"

//#include "ModelGeneratorContext.h"
//#include "rrLLVMModelDataIRBuilder.h"
//#include "rrException.h"
//#include "rrUtils.h"
//#include "rrLogger.h"

//#include "CSRMatrixTest.h"

//#include "LLVMCSRMatrixTest.h"


//#include "test_compiler.h"

//#include "TestBase.h"
//#include "TestEvalInitialConditions.h"
//#include "TestEvalReactionRates.h"
#include "TestEvalModel.h"
#include "TestRoadRunner.h"
#include "GetBoundarySpeciesAmountTest.h"
#include "TestEvalInitialConditions.h"

#include "rrRoadRunner.h"

#include "rrParameter.h"
#include "rrLogger.h"

#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/SBMLReader.h>
#include <time.h>
#include <stdio.h>




struct TestCase
{
    const char* first;
    int second;
    int events;
    int rateRule;
    int passC;
};

void getPairs(TestCase *&, int& npairs);



using namespace std;
using namespace rr;

bool RunTest(const string& version, int caseNumber);

int main(int argc, char* argv[])
{
    cout << "RoadRunner LLVM SBML Test Suite" << endl;
    cout << "built on " << __TIMESTAMP__ << endl;
    cout << rr::RoadRunner::getExtendedVersionInfo() << endl;

    const char* compiler = "llvm";

    uint a;

    std::vector<int> vec;




    Logger::enableLoggingToConsole();

    Logger::SetCutOffLogLevel(Logger::PRIO_TRACE);


    int testCase = 0;

    if (argc >= 2)
    {
        testCase = atoi(argv[1]);
        if (argc >= 3)
        {
            compiler = argv[2];
        }
    }

    Log(Logger::PRIO_NOTICE) << "running test case " << testCase;


    //runSparseTest(33, 323, 50);

    //runLLVMCSRMatrixTest(33, 323, 50);


    TestCase *pairs;
    int npairs;
    //
    //    StrIntPair test = {"l2v4", 14};
    //    runInitialValueAssigmentTest(test.first, test.second);
    //
    //    return 0;

    getPairs(pairs, npairs);

    /*

    StrIntPair p =
    { "l2v4", 190 };
    try
    {
        TestBase test(p.first, p.second);
    } catch (std::exception &e)
    {
        Log(lError) << "Error with test " << p.first << ", " << p.second
                << ": " << e.what();
    }
     */

    const int loop = 1;



    time_t start, stop;
    clock_t startc, stopc;

    /*

    startc = clock();
    time(&start);
    // Do stuff


    for (int i = 0; i < loop; ++i) {
        //runInitialValueAssigmentTest(pairs[i].first, pairs[i].second);
        try
        {
            TestRoadRunner test(pairs[0].first, pairs[0].second);
            test.test("gcc");
        }
        catch (std::exception &e)
        {
            Log(lError) << "Error with test " << pairs[i].first << ", " << pairs[i].second
                    << ": " << e.what();
        }
    }

    stopc = clock();
    time(&stop);

    printf("C Model Used %0.2f seconds of CPU time. \n", (double)(stopc - startc)/CLOCKS_PER_SEC);
    printf("C Model Finished in about %.0f seconds. \n", difftime(stop, start));

     */

    startc = clock();
    time(&start);

    int i = 2;
    //for (int i = 0; i < loop; ++i) {
    //runInitialValueAssigmentTest(pairs[i].first, pairs[i].second);
    try
    {
        TestEvalInitialConditions test(compiler, pairs[testCase].first, pairs[testCase].second);
        test.test();
        //TestRoadRunner test(compiler, pairs[testCase].first, pairs[testCase].second);
        //test.test(compiler);
        //test.saveResult();
        //test.compareReference();
    }
    catch (std::exception &e)
    {
        Log(lError) << "Error with test " << pairs[testCase].first << ", " << pairs[testCase].second
                << ": " << e.what();
    }
    //}

    stopc = clock();
    time(&stop);

    printf("LLVM Model Used %0.2f seconds of CPU time. \n", (double)(stopc - startc)/CLOCKS_PER_SEC);
    printf("LLVM Model Finished in about %.0f seconds. \n", difftime(stop, start));


    //StrIntPair test = {"l3v1", 999  };
    //StrIntPair test = {"l2v4", 7};
    //runInitialValueAssigmentTest(test.first, test.second);


    return 0;
}


/*

bool RunTest(const string& version, int caseNumber)
{
bool result = false;
try
{
string modelFileName = getModelFileName(version, caseNumber);

SBMLDocument *doc = libsbml::readSBMLFromFile(modelFileName.c_str());

        LLVMModelGeneratorContext c(doc, true);

        //StructType *s = LLVMModelDataValue::getStructType(c.getModule(), c.getExecutionEngine());

        LLVMModelDataSymbols symbols(doc->getModel(), true);

        symbols.print();




        delete doc;


    }
    catch(std::exception& ex)
    {
        string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
        return false;
    }


    return result;
}

 */






TestCase testpairs[] = {
        {"l2v4", 1, 0, 0, 1},
        {"l2v4", 2, 0, 0, 1},
        {"l2v4", 3, 0, 0, 1},
        {"l2v4", 4, 0, 0, 1},
        {"l2v4", 5, 0, 0, 1},
        {"l2v4", 6, 0, 0, 1},
        {"l2v4", 7, 0, 0, 1},
        {"l2v4", 8, 0, 0, 1},
        {"l2v4", 9, 0, 0, 1},
        {"l2v4", 10, 0, 0, 1},
        {"l2v4", 11, 0, 0, 1},
        {"l2v4", 12, 0, 0, 1},
        {"l2v4", 13, 0, 0, 1},
        {"l2v4", 14, 0, 0, 1},
        {"l2v4", 15, 0, 0, 1},
        {"l2v4", 16, 0, 0, 1},
        {"l2v4", 17, 0, 0, 1},
        {"l2v4", 18, 0, 0, 1},
        {"l2v4", 19, 0, 0, 1},
        {"l2v4", 20, 0, 0, 1},
        {"l2v4", 21, 0, 0, 1},
        {"l2v4", 22, 0, 0, 1},
        {"l2v4", 23, 0, 0, 1},
        {"l2v4", 24, 0, 0, 1},
        {"l2v4", 25, 0, 0, 1},
        {"l2v4", 26, 3, 0, 1},
        {"l2v4", 27, 0, 0, 1},
        {"l2v4", 28, 0, 0, 1},
        {"l2v4", 29, 0, 0, 1},
        {"l2v4", 30, 0, 0, 1},
        {"l2v4", 31, 0, 1, 1},
        {"l2v4", 32, 0, 1, 1},
        {"l2v4", 33, 0, 1, 1},
        {"l2v4", 34, 0, 0, 1},
        {"l2v4", 35, 0, 0, 1},
        {"l2v4", 36, 0, 0, 1},
        {"l2v4", 37, 0, 0, 1},
        {"l2v4", 38, 0, 0, 1},
        {"l2v4", 39, 0, 0, 0},
        {"l2v4", 40, 0, 1, 0},
        {"l2v4", 41, 3, 0, 1},
        {"l2v4", 42, 0, 0, 1},
        {"l2v4", 43, 0, 0, 1},
        {"l2v4", 44, 0, 0, 1},
        {"l2v4", 45, 0, 0, 1},
        {"l2v4", 46, 0, 0, 1},
        {"l2v4", 47, 0, 0, 1},
        {"l2v4", 48, 0, 0, 1},
        {"l2v4", 49, 0, 0, 1},
        {"l2v4", 50, 0, 0, 1},
        {"l2v4", 51, 0, 1, 1},
        {"l2v4", 52, 0, 1, 1},
        {"l2v4", 53, 0, 1, 1},
        {"l2v4", 54, 0, 0, 1},
        {"l2v4", 55, 0, 0, 1},
        {"l2v4", 56, 0, 0, 1},
        {"l2v4", 57, 0, 0, 1},
        {"l2v4", 58, 0, 0, 1},
        {"l2v4", 59, 0, 0, 1},
        {"l2v4", 60, 0, 0, 1},
        {"l2v4", 61, 0, 0, 1},
        {"l2v4", 62, 0, 0, 1},
        {"l2v4", 63, 0, 0, 1},
        {"l2v4", 64, 0, 0, 1},
        {"l2v4", 65, 0, 0, 1},
        {"l2v4", 66, 0, 1, 1},
        {"l2v4", 67, 0, 1, 1},
        {"l2v4", 68, 0, 0, 1},
        {"l2v4", 69, 0, 0, 1},
        {"l2v4", 70, 0, 0, 1},
        {"l2v4", 71, 3, 0, 1},
        {"l2v4", 72, 3, 0, 1},
        {"l2v4", 73, 3, 0, 1},
        {"l2v4", 74, 3, 0, 1},
        {"l2v4", 75, 0, 0, 1},
        {"l2v4", 76, 0, 0, 1},
        {"l2v4", 77, 0, 0, 1},
        {"l2v4", 78, 0, 0, 1},
        {"l2v4", 79, 0, 0, 1},
        {"l2v4", 80, 0, 0, 1},
        {"l2v4", 81, 0, 1, 1},
        {"l2v4", 82, 0, 1, 1},
        {"l2v4", 83, 0, 1, 1},
        {"l2v4", 84, 0, 1, 1},
        {"l2v4", 85, 0, 1, 1},
        {"l2v4", 86, 0, 1, 1},
        {"l2v4", 87, 0, 0, 1},
        {"l2v4", 88, 0, 0, 1},
        {"l2v4", 89, 0, 0, 1},
        {"l2v4", 90, 0, 0, 1},
        {"l2v4", 91, 0, 0, 1},
        {"l2v4", 92, 0, 1, 1},
        {"l2v4", 93, 0, 1, 1},
        {"l2v4", 94, 0, 1, 1},
        {"l2v4", 95, 0, 0, 1},
        {"l2v4", 96, 0, 0, 1},
        {"l2v4", 97, 0, 0, 1},
        {"l2v4", 98, 0, 0, 1},
        {"l2v4", 99, 0, 0, 1},
        {"l2v4", 100, 0, 0, 1},
        {"l2v4", 101, 0, 0, 1},
        {"l2v4", 102, 0, 0, 1},
        {"l2v4", 103, 0, 0, 1},
        {"l2v4", 104, 0, 1, 1},
        {"l2v4", 105, 0, 1, 1},
        {"l2v4", 106, 0, 1, 1},
        {"l2v4", 107, 0, 0, 1},
        {"l2v4", 108, 0, 0, 1},
        {"l2v4", 109, 0, 0, 1},
        {"l2v4", 110, 0, 0, 1},
        {"l2v4", 111, 0, 0, 1},
        {"l2v4", 112, 0, 0, 1},
        {"l2v4", 113, 0, 0, 1},
        {"l2v4", 114, 0, 0, 1},
        {"l2v4", 115, 0, 0, 1},
        {"l2v4", 116, 0, 0, 1},
        {"l2v4", 117, 0, 0, 1},
        {"l2v4", 118, 0, 0, 1},
        {"l2v4", 119, 0, 0, 1},
        {"l2v4", 120, 0, 0, 1},
        {"l2v4", 121, 0, 0, 1},
        {"l2v4", 122, 0, 1, 1},
        {"l2v4", 123, 0, 1, 1},
        {"l2v4", 124, 0, 1, 1},
        {"l2v4", 125, 0, 0, 1},
        {"l2v4", 126, 0, 0, 1},
        {"l2v4", 127, 0, 0, 1},
        {"l2v4", 128, 0, 0, 1},
        {"l2v4", 129, 0, 0, 1},
        {"l2v4", 130, 0, 0, 1},
        {"l2v4", 131, 0, 0, 1},
        {"l2v4", 132, 0, 0, 1},
        {"l2v4", 133, 0, 0, 1},
        {"l2v4", 134, 0, 0, 1},
        {"l2v4", 135, 0, 0, 1},
        {"l2v4", 136, 0, 0, 1},
        {"l2v4", 137, 0, 0, 1},
        {"l2v4", 138, 0, 0, 1},
        {"l2v4", 139, 0, 0, 1},
        {"l2v4", 140, 0, 0, 1},
        {"l2v4", 141, 0, 0, 1},
        {"l2v4", 142, 0, 0, 1},
        {"l2v4", 143, 0, 0, 1},
        {"l2v4", 144, 0, 0, 1},
        {"l2v4", 145, 0, 0, 1},
        {"l2v4", 146, 0, 0, 1},
        {"l2v4", 147, 0, 0, 1},
        {"l2v4", 148, 0, 0, 1},
        {"l2v4", 149, 0, 0, 1},
        {"l2v4", 150, 0, 0, 1},
        {"l2v4", 151, 0, 0, 1},
        {"l2v4", 152, 0, 0, 1},
        {"l2v4", 153, 0, 0, 1},
        {"l2v4", 154, 0, 0, 1},
        {"l2v4", 155, 0, 0, 1},
        {"l2v4", 156, 0, 0, 1},
        {"l2v4", 157, 0, 0, 1},
        {"l2v4", 158, 0, 0, 1},
        {"l2v4", 159, 0, 0, 1},
        {"l2v4", 160, 0, 0, 1},
        {"l2v4", 161, 0, 0, 1},
        {"l2v4", 162, 0, 0, 1},
        {"l2v4", 163, 0, 0, 1},
        {"l2v4", 164, 0, 0, 1},
        {"l2v4", 165, 0, 0, 1},
        {"l2v4", 166, 0, 0, 1},
        {"l2v4", 167, 0, 0, 1},
        {"l2v4", 168, 0, 0, 1},
        {"l2v4", 169, 0, 0, 1},
        {"l2v4", 170, 0, 0, 1},
        {"l2v4", 171, 0, 0, 1},
        {"l2v4", 172, 3, 0, 1},
        {"l2v4", 173, 0, 0, 1},
        {"l2v4", 174, 0, 0, 1},
        {"l2v4", 175, 0, 0, 1},
        {"l2v4", 176, 0, 0, 1},
        {"l2v4", 177, 0, 0, 1},
        {"l2v4", 178, 0, 0, 1},
        {"l2v4", 179, 0, 0, 1},
        {"l2v4", 180, 0, 0, 1},
        {"l2v4", 181, 0, 0, 1},
        {"l2v4", 183, 0, 0, 1},
        {"l2v4", 185, 0, 0, 1},
        {"l2v4", 186, 0, 0, 1},
        {"l2v4", 187, 0, 0, 1},
        {"l2v4", 188, 0, 0, 1},
        {"l2v4", 189, 0, 0, 1},
        {"l2v4", 190, 0, 0, 1},
        {"l2v4", 191, 0, 0, 1},
        {"l2v4", 192, 0, 0, 1},
        {"l2v4", 193, 0, 0, 1},
        {"l2v4", 194, 0, 0, 1},
        {"l2v4", 195, 0, 0, 1},
        {"l2v4", 196, 0, 0, 1},
        {"l2v4", 197, 0, 0, 1},
        {"l2v4", 198, 0, 0, 1},
        {"l2v4", 199, 0, 0, 1},
        {"l2v4", 200, 0, 0, 1},
        {"l2v4", 201, 0, 0, 1},
        {"l2v4", 202, 0, 0, 1},
        {"l2v4", 203, 0, 0, 1},
        {"l2v4", 204, 0, 0, 1},
        {"l2v4", 205, 0, 0, 1},
        {"l2v4", 206, 0, 0, 1},
        {"l2v4", 207, 0, 0, 1},
        {"l2v4", 208, 0, 0, 1},
        {"l2v4", 209, 0, 0, 1},
        {"l2v4", 210, 0, 0, 1},
        {"l2v4", 211, 0, 0, 1},
        {"l2v4", 212, 0, 0, 1},
        {"l2v4", 213, 0, 0, 1},
        {"l2v4", 214, 0, 0, 1},
        {"l2v4", 215, 0, 0, 1},
        {"l2v4", 216, 0, 0, 1},
        {"l2v4", 217, 0, 0, 1},
        {"l2v4", 218, 0, 0, 1},
        {"l2v4", 219, 0, 0, 1},
        {"l2v4", 220, 0, 0, 1},
        {"l2v4", 221, 0, 0, 1},
        {"l2v4", 222, 0, 0, 1},
        {"l2v4", 223, 0, 0, 1},
        {"l2v4", 224, 0, 0, 1},
        {"l2v4", 225, 0, 0, 1},
        {"l2v4", 226, 0, 0, 1},
        {"l2v4", 227, 0, 0, 1},
        {"l2v4", 228, 0, 0, 1},
        {"l2v4", 229, 0, 0, 1},
        {"l2v4", 230, 0, 0, 1},
        {"l2v4", 231, 0, 0, 1},
        {"l2v4", 232, 0, 0, 1},
        {"l2v4", 233, 0, 0, 1},
        {"l2v4", 234, 0, 0, 1},
        {"l2v4", 235, 0, 0, 1},
        {"l2v4", 236, 0, 0, 1},
        {"l2v4", 237, 0, 0, 1},
        {"l2v4", 238, 0, 0, 1},
        {"l2v4", 239, 0, 0, 1},
        {"l2v4", 240, 0, 0, 1},
        {"l2v4", 241, 0, 0, 1},
        {"l2v4", 242, 0, 0, 1},
        {"l2v4", 243, 0, 0, 1},
        {"l2v4", 244, 0, 0, 1},
        {"l2v4", 245, 0, 0, 1},
        {"l2v4", 246, 0, 0, 1},
        {"l2v4", 247, 0, 0, 1},
        {"l2v4", 248, 0, 0, 1},
        {"l2v4", 249, 0, 0, 1},
        {"l2v4", 250, 0, 0, 1},
        {"l2v4", 251, 0, 0, 1},
        {"l2v4", 252, 0, 0, 1},
        {"l2v4", 253, 0, 0, 1},
        {"l2v4", 254, 0, 0, 1},
        {"l2v4", 255, 0, 0, 1},
        {"l2v4", 256, 0, 0, 1},
        {"l2v4", 257, 0, 0, 1},
        {"l2v4", 258, 0, 0, 1},
        {"l2v4", 259, 0, 0, 1},
        {"l2v4", 260, 0, 0, 1},
        {"l2v4", 261, 0, 0, 1},
        {"l2v4", 262, 0, 0, 1},
        {"l2v4", 263, 0, 0, 1},
        {"l2v4", 264, 0, 0, 1},
        {"l2v4", 265, 0, 0, 1},
        {"l2v4", 266, 0, 0, 1},
        {"l2v4", 267, 0, 0, 1},
        {"l2v4", 268, 0, 0, 1},
        {"l2v4", 269, 0, 0, 1},
        {"l2v4", 270, 0, 0, 1},
        {"l2v4", 271, 0, 0, 1},
        {"l2v4", 272, 0, 0, 1},
        {"l2v4", 273, 0, 0, 1},
        {"l2v4", 274, 0, 0, 1},
        {"l2v4", 275, 0, 0, 1},
        {"l2v4", 276, 0, 0, 1},
        {"l2v4", 277, 0, 0, 1},
        {"l2v4", 278, 0, 0, 1},
        {"l2v4", 279, 0, 0, 1},
        {"l2v4", 280, 0, 0, 1},
        {"l2v4", 281, 0, 0, 1},
        {"l2v4", 282, 0, 0, 1},
        {"l2v4", 283, 0, 0, 1},
        {"l2v4", 284, 0, 0, 1},
        {"l2v4", 285, 0, 0, 1},
        {"l2v4", 286, 0, 0, 1},
        {"l2v4", 287, 0, 0, 1},
        {"l2v4", 288, 0, 0, 1},
        {"l2v4", 289, 0, 0, 1},
        {"l2v4", 290, 0, 0, 1},
        {"l2v4", 291, 0, 0, 1},
        {"l2v4", 292, 0, 0, 1},
        {"l2v4", 293, 0, 0, 1},
        {"l2v4", 294, 0, 0, 1},
        {"l2v4", 295, 0, 0, 1},
        {"l2v4", 296, 0, 0, 1},
        {"l2v4", 297, 0, 0, 1},
        {"l2v4", 298, 0, 0, 1},
        {"l2v4", 299, 0, 0, 1},
        {"l2v4", 300, 0, 0, 1},
        {"l2v4", 301, 0, 0, 1},
        {"l2v4", 302, 0, 0, 1},
        {"l2v4", 303, 0, 0, 1},
        {"l2v4", 304, 0, 0, 1},
        {"l2v4", 305, 0, 0, 1},
        {"l2v4", 306, 0, 0, 1},
        {"l2v4", 307, 0, 0, 1},
        {"l2v4", 308, 0, 0, 1},
        {"l2v4", 309, 0, 0, 1},
        {"l2v4", 310, 0, 0, 1},
        {"l2v4", 311, 0, 0, 1},
        {"l2v4", 312, 0, 0, 1},
        {"l2v4", 313, 0, 0, 1},
        {"l2v4", 314, 0, 0, 1},
        {"l2v4", 315, 0, 0, 1},
        {"l2v4", 316, 0, 0, 1},
        {"l2v4", 317, 0, 0, 1},
        {"l2v4", 318, 0, 0, 1},
        {"l2v4", 319, 0, 0, 1},
        {"l2v4", 320, 0, 0, 1},
        {"l2v4", 321, 0, 0, 1},
        {"l2v4", 322, 0, 0, 1},
        {"l2v4", 323, 0, 0, 1},
        {"l2v4", 324, 0, 0, 1},
        {"l2v4", 325, 0, 0, 1},
        {"l2v4", 326, 0, 0, 1},
        {"l2v4", 327, 0, 0, 1},
        {"l2v4", 328, 0, 0, 1},
        {"l2v4", 329, 0, 0, 1},
        {"l2v4", 330, 0, 0, 1},
        {"l2v4", 331, 0, 0, 1},
        {"l2v4", 332, 0, 0, 1},
        {"l2v4", 333, 0, 0, 1},
        {"l2v4", 334, 0, 0, 1},
        {"l2v4", 335, 0, 0, 1},
        {"l2v4", 336, 0, 0, 1},
        {"l2v4", 337, 0, 0, 1},
        {"l2v4", 338, 0, 0, 1},
        {"l2v4", 339, 0, 0, 1},
        {"l2v4", 340, 0, 0, 1},
        {"l2v4", 341, 0, 0, 1},
        {"l2v4", 342, 0, 0, 1},
        {"l2v4", 343, 0, 0, 1},
        {"l2v4", 344, 0, 0, 1},
        {"l2v4", 345, 0, 0, 1},
        {"l2v4", 346, 0, 0, 1},
        {"l2v4", 347, 0, 0, 1},
        {"l2v4", 348, 3, 0, 1},
        {"l2v4", 349, 3, 0, 1},
        {"l2v4", 350, 3, 0, 1},
        {"l2v4", 351, 3, 0, 1},
        {"l2v4", 352, 3, 0, 1},
        {"l2v4", 353, 3, 0, 1},
        {"l2v4", 354, 3, 0, 1},
        {"l2v4", 355, 3, 0, 1},
        {"l2v4", 356, 3, 0, 1},
        {"l2v4", 357, 3, 0, 1},
        {"l2v4", 358, 3, 0, 1},
        {"l2v4", 359, 3, 0, 1},
        {"l2v4", 360, 3, 0, 1},
        {"l2v4", 361, 3, 0, 1},
        {"l2v4", 362, 3, 0, 1},
        {"l2v4", 363, 3, 0, 1},
        {"l2v4", 364, 3, 0, 1},
        {"l2v4", 365, 3, 0, 1},
        {"l2v4", 366, 3, 0, 1},
        {"l2v4", 367, 3, 0, 1},
        {"l2v4", 368, 3, 0, 1},
        {"l2v4", 369, 3, 0, 1},
        {"l2v4", 370, 3, 0, 1},
        {"l2v4", 371, 3, 0, 1},
        {"l2v4", 372, 3, 0, 1},
        {"l2v4", 373, 3, 0, 1},
        {"l2v4", 374, 3, 0, 1},
        {"l2v4", 375, 3, 0, 1},
        {"l2v4", 376, 3, 0, 1},
        {"l2v4", 377, 3, 0, 1},
        {"l2v4", 378, 3, 0, 1},
        {"l2v4", 379, 3, 0, 1},
        {"l2v4", 380, 3, 0, 1},
        {"l2v4", 381, 3, 0, 1},
        {"l2v4", 382, 3, 0, 1},
        {"l2v4", 383, 3, 0, 1},
        {"l2v4", 384, 3, 0, 1},
        {"l2v4", 385, 3, 0, 1},
        {"l2v4", 386, 3, 0, 1},
        {"l2v4", 387, 3, 0, 1},
        {"l2v4", 388, 3, 0, 1},
        {"l2v4", 389, 3, 0, 1},
        {"l2v4", 390, 3, 0, 1},
        {"l2v4", 391, 3, 0, 1},
        {"l2v4", 392, 3, 0, 1},
        {"l2v4", 393, 3, 0, 1},
        {"l2v4", 394, 3, 0, 1},
        {"l2v4", 395, 3, 0, 1},
        {"l2v4", 396, 3, 0, 1},
        {"l2v4", 397, 3, 0, 1},
        {"l2v4", 398, 3, 0, 1},
        {"l2v4", 399, 3, 0, 1},
        {"l2v4", 400, 3, 0, 1},
        {"l2v4", 401, 3, 0, 1},
        {"l2v4", 402, 3, 0, 1},
        {"l2v4", 403, 3, 0, 1},
        {"l2v4", 404, 3, 0, 1},
        {"l2v4", 405, 3, 0, 1},
        {"l2v4", 406, 3, 0, 1},
        {"l2v4", 407, 3, 0, 1},
        {"l2v4", 408, 3, 0, 1},
        {"l2v4", 409, 3, 0, 1},
        {"l2v4", 410, 3, 0, 1},
        {"l2v4", 411, 3, 0, 1},
        {"l2v4", 412, 3, 0, 1},
        {"l2v4", 413, 3, 0, 1},
        {"l2v4", 414, 3, 0, 1},
        {"l2v4", 415, 3, 0, 1},
        {"l2v4", 416, 3, 0, 1},
        {"l2v4", 417, 3, 0, 1},
        {"l2v4", 418, 3, 0, 1},
        {"l2v4", 419, 3, 0, 1},
        {"l2v4", 420, 3, 0, 1},
        {"l2v4", 421, 3, 0, 1},
        {"l2v4", 422, 3, 0, 1},
        {"l2v4", 423, 3, 0, 1},
        {"l2v4", 424, 3, 0, 1},
        {"l2v4", 425, 3, 0, 1},
        {"l2v4", 426, 3, 0, 1},
        {"l2v4", 427, 3, 0, 1},
        {"l2v4", 428, 3, 0, 1},
        {"l2v4", 429, 3, 0, 1},
        {"l2v4", 430, 3, 0, 1},
        {"l2v4", 431, 3, 0, 1},
        {"l2v4", 432, 3, 0, 1},
        {"l2v4", 433, 3, 0, 1},
        {"l2v4", 434, 3, 0, 1},
        {"l2v4", 435, 3, 0, 1},
        {"l2v4", 436, 3, 0, 1},
        {"l2v4", 437, 3, 0, 1},
        {"l2v4", 438, 3, 0, 1},
        {"l2v4", 439, 3, 0, 1},
        {"l2v4", 440, 3, 0, 1},
        {"l2v4", 441, 3, 0, 1},
        {"l2v4", 442, 3, 0, 1},
        {"l2v4", 443, 3, 0, 1},
        {"l2v4", 444, 3, 0, 1},
        {"l2v4", 445, 3, 0, 1},
        {"l2v4", 446, 3, 0, 1},
        {"l2v4", 447, 3, 0, 1},
        {"l2v4", 448, 3, 0, 1},
        {"l2v4", 449, 3, 0, 1},
        {"l2v4", 450, 3, 0, 1},
        {"l2v4", 451, 3, 0, 1},
        {"l2v4", 452, 3, 0, 1},
        {"l2v4", 453, 3, 0, 1},
        {"l2v4", 454, 3, 0, 1},
        {"l2v4", 455, 3, 0, 1},
        {"l2v4", 456, 3, 0, 1},
        {"l2v4", 457, 3, 0, 1},
        {"l2v4", 458, 3, 0, 1},
        {"l2v4", 459, 3, 0, 1},
        {"l2v4", 460, 3, 0, 1},
        {"l2v4", 461, 3, 0, 1},
        {"l2v4", 462, 0, 0, 1},
        {"l2v4", 463, 0, 0, 1},
        {"l2v4", 464, 0, 0, 1},
        {"l2v4", 465, 0, 0, 1},
        {"l2v4", 466, 0, 0, 1},
        {"l2v4", 467, 0, 0, 1},
        {"l2v4", 468, 0, 0, 1},
        {"l2v4", 469, 0, 0, 1},
        {"l2v4", 470, 0, 0, 1},
        {"l2v4", 471, 0, 0, 1},
        {"l2v4", 472, 0, 0, 1},
        {"l2v4", 473, 0, 0, 1},
        {"l2v4", 474, 0, 0, 1},
        {"l2v4", 475, 0, 0, 1},
        {"l2v4", 476, 0, 0, 1},
        {"l2v4", 477, 0, 0, 1},
        {"l2v4", 478, 0, 0, 1},
        {"l2v4", 479, 0, 0, 1},
        {"l2v4", 480, 0, 0, 1},
        {"l2v4", 481, 0, 0, 1},
        {"l2v4", 482, 0, 0, 1},
        {"l2v4", 483, 0, 0, 1},
        {"l2v4", 484, 0, 0, 1},
        {"l2v4", 485, 0, 0, 1},
        {"l2v4", 486, 0, 0, 1},
        {"l2v4", 487, 0, 0, 1},
        {"l2v4", 488, 0, 0, 1},
        {"l2v4", 489, 0, 0, 1},
        {"l2v4", 490, 0, 0, 1},
        {"l2v4", 491, 0, 0, 1},
        {"l2v4", 492, 0, 0, 1},
        {"l2v4", 493, 0, 0, 1},
        {"l2v4", 494, 0, 0, 1},
        {"l2v4", 495, 0, 0, 1},
        {"l2v4", 496, 0, 0, 1},
        {"l2v4", 497, 0, 0, 1},
        {"l2v4", 498, 0, 0, 1},
        {"l2v4", 499, 0, 0, 1},
        {"l2v4", 500, 0, 0, 1},
        {"l2v4", 501, 0, 0, 1},
        {"l2v4", 502, 0, 0, 1},
        {"l2v4", 503, 0, 0, 1},
        {"l2v4", 504, 0, 0, 1},
        {"l2v4", 505, 0, 0, 1},
        {"l2v4", 506, 0, 0, 1},
        {"l2v4", 507, 0, 0, 1},
        {"l2v4", 508, 0, 0, 1},
        {"l2v4", 509, 0, 0, 1},
        {"l2v4", 510, 0, 0, 1},
        {"l2v4", 511, 0, 0, 1},
        {"l2v4", 512, 0, 0, 1},
        {"l2v4", 513, 0, 0, 1},
        {"l2v4", 514, 0, 0, 1},
        {"l2v4", 515, 0, 0, 1},
        {"l2v4", 516, 0, 0, 1},
        {"l2v4", 517, 0, 0, 1},
        {"l2v4", 518, 0, 0, 1},
        {"l2v4", 519, 0, 0, 1},
        {"l2v4", 520, 0, 0, 1},
        {"l2v4", 521, 0, 0, 1},
        {"l2v4", 522, 0, 0, 1},
        {"l2v4", 523, 0, 0, 1},
        {"l2v4", 524, 0, 0, 1},
        {"l2v4", 525, 0, 0, 1},
        {"l2v4", 526, 0, 0, 1},
        {"l2v4", 527, 0, 0, 1},
        {"l2v4", 528, 0, 0, 1},
        {"l2v4", 529, 0, 0, 1},
        {"l2v4", 530, 0, 0, 1},
        {"l2v4", 532, 0, 0, 1},
        {"l2v4", 539, 0, 0, 1},
        {"l2v4", 540, 0, 0, 1},
        {"l2v4", 541, 0, 0, 1},
        {"l2v4", 542, 0, 0, 1},
        {"l2v4", 544, 0, 0, 1},
        {"l2v4", 545, 0, 0, 1},
        {"l2v4", 547, 0, 0, 1},
        {"l2v4", 568, 0, 0, 1},
        {"l2v4", 572, 0, 0, 1},
        {"l2v4", 574, 0, 0, 1},
        {"l2v4", 577, 0, 0, 1},
        {"l2v4", 578, 0, 0, 1},
        {"l2v4", 579, 0, 0, 1},
        {"l2v4", 580, 0, 0, 1},
        {"l2v4", 581, 0, 0, 1},
        {"l2v4", 582, 0, 0, 1},
        {"l2v4", 583, 0, 0, 1},
        {"l2v4", 584, 0, 0, 1},
        {"l2v4", 585, 0, 0, 1},
        {"l2v4", 586, 0, 0, 1},
        {"l2v4", 587, 0, 0, 1},
        {"l2v4", 588, 0, 0, 1},
        {"l2v4", 589, 0, 0, 1},
        {"l2v4", 590, 0, 0, 1},
        {"l2v4", 591, 0, 0, 1},
        {"l2v4", 592, 0, 0, 1},
        {"l2v4", 593, 0, 0, 1},
        {"l2v4", 594, 0, 0, 1},
        {"l2v4", 595, 0, 0, 1},
        {"l2v4", 596, 0, 0, 1},
        {"l2v4", 597, 0, 0, 1},
        {"l2v4", 598, 0, 0, 1},
        {"l2v4", 599, 0, 0, 1},
        {"l2v4", 600, 0, 0, 1},
        {"l2v4", 601, 0, 0, 1},
        {"l2v4", 602, 0, 0, 1},
        {"l2v4", 603, 0, 0, 1},
        {"l2v4", 604, 0, 0, 1},
        {"l2v4", 605, 0, 0, 1},
        {"l2v4", 606, 0, 0, 1},
        {"l2v4", 607, 0, 0, 1},
        {"l2v4", 608, 0, 0, 1},
        {"l2v4", 609, 0, 0, 1},
        {"l2v4", 610, 0, 0, 1},
        {"l2v4", 611, 0, 0, 1},
        {"l2v4", 612, 0, 0, 1},
        {"l2v4", 616, 0, 0, 1},
        {"l2v4", 617, 0, 0, 1},
        {"l2v4", 618, 0, 0, 1},
        {"l2v4", 619, 3, 0, 1},
        {"l2v4", 620, 3, 0, 1},
        {"l2v4", 621, 3, 0, 1},
        {"l2v4", 622, 3, 0, 1},
        {"l2v4", 623, 3, 0, 1},
        {"l2v4", 624, 3, 0, 1},
        {"l2v4", 625, 0, 0, 1},
        {"l2v4", 626, 0, 0, 1},
        {"l2v4", 627, 0, 0, 1},
        {"l2v4", 631, 0, 0, 1},
        {"l2v4", 632, 0, 0, 1},
        {"l2v4", 633, 0, 0, 1},
        {"l2v4", 634, 3, 0, 1},
        {"l2v4", 635, 3, 0, 1},
        {"l2v4", 636, 3, 0, 1},
        {"l2v4", 637, 3, 0, 1},
        {"l2v4", 638, 3, 0, 1},
        {"l2v4", 639, 3, 0, 1},
        {"l2v4", 640, 0, 0, 1},
        {"l2v4", 641, 0, 0, 1},
        {"l2v4", 642, 0, 0, 1},
        {"l2v4", 643, 0, 0, 1},
        {"l2v4", 644, 0, 0, 1},
        {"l2v4", 645, 0, 0, 1},
        {"l2v4", 646, 3, 0, 1},
        {"l2v4", 647, 3, 0, 1},
        {"l2v4", 648, 3, 0, 1},
        {"l2v4", 649, 3, 0, 1},
        {"l2v4", 650, 3, 0, 1},
        {"l2v4", 651, 3, 0, 1},
        {"l2v4", 652, 3, 0, 1},
        {"l2v4", 653, 3, 0, 1},
        {"l2v4", 654, 3, 0, 1},
        {"l2v4", 655, 3, 0, 1},
        {"l2v4", 656, 3, 0, 1},
        {"l2v4", 657, 3, 0, 1},
        {"l2v4", 667, 0, 0, 1},
        {"l2v4", 668, 0, 0, 1},
        {"l2v4", 669, 0, 0, 1},
        {"l2v4", 670, 0, 0, 1},
        {"l2v4", 671, 0, 0, 1},
        {"l2v4", 672, 0, 0, 1},
        {"l2v4", 676, 0, 0, 1},
        {"l2v4", 677, 0, 0, 1},
        {"l2v4", 678, 0, 0, 1},
        {"l2v4", 679, 3, 0, 1},
        {"l2v4", 680, 3, 0, 1},
        {"l2v4", 681, 3, 0, 1},
        {"l2v4", 682, 3, 0, 1},
        {"l2v4", 683, 3, 0, 1},
        {"l2v4", 684, 3, 0, 1},
        {"l2v4", 685, 0, 0, 1},
        {"l2v4", 686, 0, 0, 1},
        {"l2v4", 688, 0, 0, 1},
        {"l2v4", 689, 3, 0, 1},
        {"l2v4", 690, 3, 0, 1},
        {"l2v4", 691, 0, 0, 1},
        {"l2v4", 692, 0, 0, 1},
        {"l2v4", 693, 0, 0, 1},
        {"l2v4", 694, 0, 0, 1},
        {"l2v4", 697, 0, 0, 1},
        {"l2v4", 698, 0, 0, 1},
        {"l2v4", 699, 3, 0, 1},
        {"l2v4", 700, 3, 0, 1},
        {"l2v4", 701, 3, 0, 1},
        {"l2v4", 702, 3, 0, 1},
        {"l2v4", 703, 0, 0, 1},
        {"l2v4", 704, 0, 0, 1},
        {"l2v4", 706, 0, 0, 1},
        {"l2v4", 707, 3, 0, 1},
        {"l2v4", 708, 3, 0, 1},
        {"l2v4", 709, 0, 0, 1},
        {"l2v4", 710, 0, 0, 1},
        {"l2v4", 711, 0, 0, 1},
        {"l2v4", 712, 0, 0, 1},
        {"l2v4", 713, 0, 0, 1},
        {"l2v4", 714, 0, 0, 1},
        {"l2v4", 715, 0, 0, 1},
        {"l2v4", 716, 0, 0, 1},
        {"l2v4", 717, 0, 0, 1},
        {"l2v4", 718, 0, 0, 1},
        {"l2v4", 719, 0, 0, 1},
        {"l2v4", 720, 0, 0, 1},
        {"l2v4", 721, 0, 0, 1},
        {"l2v4", 722, 0, 0, 1},
        {"l2v4", 723, 3, 0, 1},
        {"l2v4", 724, 3, 0, 1},
        {"l2v4", 725, 0, 0, 1},
        {"l2v4", 726, 0, 0, 1},
        {"l2v4", 727, 0, 0, 1},
        {"l2v4", 728, 0, 0, 1},
        {"l2v4", 729, 0, 0, 1},
        {"l2v4", 730, 3, 0, 1},
        {"l2v4", 731, 3, 0, 1},
        {"l2v4", 732, 0, 0, 1},
        {"l2v4", 733, 0, 0, 1},
        {"l2v4", 734, 0, 0, 1},
        {"l2v4", 735, 0, 0, 1},
        {"l2v4", 736, 3, 0, 1},
        {"l2v4", 737, 3, 0, 1},
        {"l2v4", 738, 0, 0, 1},
        {"l2v4", 739, 0, 0, 1},
        {"l2v4", 740, 0, 0, 1},
        {"l2v4", 741, 0, 0, 1},
        {"l2v4", 742, 0, 0, 1},
        {"l2v4", 743, 3, 0, 1},
        {"l2v4", 744, 3, 0, 1},
        {"l2v4", 745, 3, 0, 1},
        {"l2v4", 746, 3, 0, 1},
        {"l2v4", 747, 3, 0, 1},
        {"l2v4", 748, 3, 0, 1},
        {"l2v4", 749, 3, 0, 1},
        {"l2v4", 750, 3, 0, 1},
        {"l2v4", 751, 3, 0, 1},
        {"l2v4", 752, 3, 0, 1},
        {"l2v4", 753, 3, 0, 1},
        {"l2v4", 754, 3, 0, 1},
        {"l2v4", 755, 3, 0, 1},
        {"l2v4", 756, 3, 0, 1},
        {"l2v4", 757, 3, 0, 1},
        {"l2v4", 758, 3, 0, 1},
        {"l2v4", 759, 3, 0, 1},
        {"l2v4", 763, 3, 0, 1},
        {"l2v4", 764, 3, 0, 1},
        {"l2v4", 765, 3, 0, 1},
        {"l2v4", 766, 3, 0, 1},
        {"l2v4", 767, 3, 0, 1},
        {"l2v4", 768, 3, 0, 1},
        {"l2v4", 769, 3, 0, 1},
        {"l2v4", 770, 3, 0, 1},
        {"l2v4", 771, 3, 0, 1},
        {"l2v4", 772, 3, 0, 1},
        {"l2v4", 773, 3, 0, 1},
        {"l2v4", 774, 3, 0, 1},
        {"l2v4", 775, 3, 0, 1},
        {"l2v4", 776, 3, 0, 1},
        {"l2v4", 781, 0, 0, 1},
        {"l2v4", 782, 0, 0, 1},
        {"l2v4", 783, 0, 0, 1},
        {"l2v4", 784, 0, 0, 1},
        {"l2v4", 785, 0, 0, 1},
        {"l2v4", 786, 0, 0, 1},
        {"l2v4", 787, 0, 0, 1},
        {"l2v4", 788, 0, 0, 1},
        {"l2v4", 789, 3, 0, 1},
        {"l2v4", 790, 3, 0, 1},
        {"l2v4", 791, 3, 0, 1},
        {"l2v4", 792, 0, 0, 1},
        {"l2v4", 793, 0, 0, 1},
        {"l2v4", 794, 0, 0, 1},
        {"l2v4", 795, 0, 0, 1},
        {"l2v4", 796, 0, 0, 1},
        {"l2v4", 797, 0, 0, 1},
        {"l2v4", 798, 0, 0, 1},
        {"l2v4", 799, 0, 0, 1},
        {"l2v4", 800, 0, 0, 1},
        {"l2v4", 801, 0, 0, 1},
        {"l2v4", 802, 0, 0, 1},
        {"l2v4", 803, 0, 0, 1},
        {"l2v4", 804, 0, 0, 1},
        {"l2v4", 805, 0, 0, 1},
        {"l2v4", 806, 0, 0, 1},
        {"l2v4", 807, 0, 0, 1},
        {"l2v4", 808, 0, 0, 1},
        {"l2v4", 809, 0, 0, 1},
        {"l2v4", 810, 0, 0, 1},
        {"l2v4", 811, 0, 0, 1},
        {"l2v4", 812, 0, 0, 1},
        {"l2v4", 813, 0, 0, 1},
        {"l2v4", 814, 0, 0, 1},
        {"l2v4", 815, 0, 0, 1},
        {"l2v4", 816, 0, 0, 1},
        {"l2v4", 817, 0, 0, 1},
        {"l2v4", 818, 0, 0, 1},
        {"l2v4", 819, 0, 0, 1},
        {"l2v4", 820, 0, 0, 1},
        {"l2v4", 821, 0, 0, 1},
        {"l2v4", 822, 0, 0, 1},
        {"l2v4", 823, 0, 0, 1},
        {"l2v4", 824, 0, 0, 1},
        {"l2v4", 825, 0, 0, 1},
        {"l2v4", 826, 0, 0, 1},
        {"l2v4", 827, 0, 0, 1},
        {"l2v4", 828, 0, 0, 1},
        {"l2v4", 829, 0, 0, 1},
        {"l2v4", 830, 0, 0, 1},
        {"l2v4", 831, 0, 0, 1},
        {"l2v4", 832, 0, 0, 1},
        {"l2v4", 833, 0, 0, 1},
        {"l2v4", 834, 0, 0, 1},
        {"l2v4", 835, 0, 0, 1},
        {"l2v4", 836, 0, 0, 1},
        {"l2v4", 837, 0, 0, 1},
        {"l2v4", 838, 0, 0, 1},
        {"l2v4", 839, 0, 0, 1},
        {"l2v4", 840, 0, 0, 1},
        {"l2v4", 841, 0, 0, 1},
        {"l2v4", 842, 0, 0, 1},
        {"l2v4", 843, 0, 0, 1},
        {"l2v4", 845, 3, 0, 1},
        {"l2v4", 846, 3, 0, 1},
        {"l2v4", 847, 3, 0, 1},
        {"l2v4", 848, 3, 0, 1},
        {"l2v4", 849, 3, 0, 1},
        {"l2v4", 850, 3, 0, 1},
        {"l2v4", 851, 0, 0, 1},
        {"l2v4", 852, 0, 0, 1},
        {"l2v4", 853, 0, 0, 1},
        {"l2v4", 854, 0, 0, 1},
        {"l2v4", 855, 0, 0, 1},
        {"l2v4", 856, 0, 0, 1},
        {"l2v4", 857, 0, 0, 1},
        {"l2v4", 858, 0, 0, 1},
        {"l2v4", 859, 0, 0, 1},
        {"l2v4", 860, 0, 0, 1},
        {"l2v4", 861, 0, 0, 1},
        {"l2v4", 862, 0, 0, 1},
        {"l2v4", 863, 0, 0, 1},
        {"l2v4", 864, 0, 0, 1},
        {"l2v4", 865, 0, 0, 1},
        {"l2v4", 866, 0, 0, 1},
        {"l2v4", 867, 0, 0, 1},
        {"l2v4", 868, 0, 0, 1},
        {"l2v4", 869, 0, 0, 1},
        {"l2v4", 877, 0, 0, 1},
        {"l2v4", 878, 0, 0, 1},
        {"l2v4", 879, 0, 0, 1},
        {"l2v4", 880, 0, 0, 1},
        {"l2v4", 881, 0, 0, 1},
        {"l2v4", 882, 0, 0, 1},
        {"l2v4", 883, 3, 0, 1},
        {"l2v4", 884, 3, 0, 1},
        {"l2v4", 885, 3, 0, 1},
        {"l2v4", 886, 3, 0, 1},
        {"l2v4", 887, 3, 0, 1},
        {"l2v4", 888, 0, 0, 1},
        {"l2v4", 889, 0, 0, 1},
        {"l2v4", 890, 0, 0, 1},
        {"l2v4", 891, 0, 0, 1},
        {"l2v4", 892, 0, 0, 1},
        {"l2v4", 893, 0, 0, 1},
        {"l2v4", 894, 0, 0, 1},
        {"l2v4", 895, 0, 0, 1},
        {"l2v4", 896, 0, 0, 1},
        {"l2v4", 897, 0, 0, 1},
        {"l2v4", 898, 0, 0, 1},
        {"l2v4", 899, 0, 0, 1},
        {"l2v4", 900, 0, 0, 1},
        {"l2v4", 901, 0, 0, 1},
        {"l2v4", 902, 0, 0, 1},
        {"l2v4", 903, 0, 0, 1},
        {"l2v4", 904, 0, 0, 1},
        {"l2v4", 905, 0, 0, 1},
        {"l2v4", 906, 0, 0, 1},
        {"l2v4", 907, 0, 0, 1},
        {"l2v4", 908, 0, 0, 1},
        {"l2v4", 909, 0, 0, 1},
        {"l2v4", 910, 0, 0, 1},
        {"l2v4", 911, 0, 0, 1},
        {"l2v4", 912, 0, 0, 1},
        {"l2v4", 913, 0, 0, 1},
        {"l2v4", 914, 0, 0, 1},
        {"l2v4", 915, 0, 0, 1},
        {"l2v4", 916, 0, 0, 1},
        {"l2v4", 917, 0, 0, 1},
        {"l2v4", 918, 0, 0, 1},
        {"l2v4", 919, 0, 0, 1},
        {"l2v4", 920, 0, 0, 1},
        {"l2v4", 921, 0, 0, 1},
        {"l2v4", 922, 0, 0, 1},
        {"l2v4", 923, 0, 0, 1},
        {"l2v4", 924, 0, 0, 1},
        {"l2v4", 925, 0, 0, 1},
        {"l2v4", 926, 0, 0, 1},
        {"l2v4", 927, 0, 0, 1},
        {"l3v1", 928, 3, 0, 1},
        {"l3v1", 929, 3, 0, 1},
        {"l3v1", 930, 3, 0, 1},
        {"l3v1", 931, 3, 0, 1},
        {"l3v1", 932, 3, 0, 1},
        {"l3v1", 933, 1, 0, 0},
        {"l3v1", 934, 3, 0, 1},
        {"l3v1", 935, 3, 0, 1},
        {"l2v4", 936, 3, 0, 1},
        {"l2v4", 944, 3, 0, 1},
        {"l2v4", 945, 3, 0, 1},
        {"l2v4", 946, 3, 0, 1},
        {"l2v4", 947, 3, 0, 1},
        {"l2v4", 948, 3, 0, 1},
        {"l2v4", 949, 0, 0, 1},
        {"l3v1", 952, 3, 0, 1},
        {"l3v1", 953, 3, 0, 1},
        {"l2v4", 954, 0, 0, 1},
        {"l2v4", 956, 0, 0, 1},
        {"l3v1", 960, 0, 0, 1},
        {"l3v1", 961, 0, 0, 1},
        {"l3v1", 963, 3, 0, 1},
        {"l3v1", 964, 3, 0, 1},
        {"l3v1", 967, 3, 0, 1},
        {"l2v4", 968, 0, 0, 1},
        {"l3v1", 969, 0, 0, 1},
        {"l3v1", 970, 0, 0, 1},
        {"l3v1", 971, 0, 0, 1},
        {"l3v1", 972, 3, 0, 1},
        {"l2v4", 973, 0, 0, 1},
        {"l3v1", 974, 0, 0, 1},
        {"l3v1", 975, 0, 0, 1},
        {"l3v1", 976, 0, 0, 1},
        {"l3v1", 977, 0, 0, 1},
        {"l2v4", 979, 3, 0, 1},
        {"l3v1", 980, 3, 0, 1},
        {"l2v4", 989, 0, 0, 1},
        {"l2v4", 990, 0, 0, 1},
        {"l2v4", 991, 3, 0, 1},
        {"l2v4", 992, 0, 0, 1},
        {"l2v4", 994, 0, 0, 1},
        {"l3v1", 998, 0, 0, 1},
        {"l3v1", 999, 0, 0, 1},
        {"l3v1", 1001, 0, 0, 1},
        {"l3v1", 1002, 0, 0, 1},
        {"l3v1", 1003, 0, 0, 1},
        {"l3v1", 1004, 0, 0, 1},
        {"l3v1", 1005, 0, 0, 1},
        {"l3v1", 1006, 0, 0, 1},
        {"l3v1", 1007, 0, 0, 1},
        {"l3v1", 1008, 0, 0, 1},
        {"l3v1", 1009, 0, 0, 1},
        {"l3v1", 1010, 0, 0, 1},
        {"l3v1", 1011, 0, 0, 1},
        {"l3v1", 1012, 0, 0, 1},
        {"l3v1", 1013, 0, 0, 1},
        {"l3v1", 1014, 0, 0, 1},
        {"l3v1", 1015, 0, 0, 1},
        {"l3v1", 1016, 0, 0, 1},
        {"l3v1", 1017, 0, 0, 1},
        {"l3v1", 1018, 0, 0, 1},
        {"l3v1", 1019, 0, 0, 1},
        {"l3v1", 1020, 0, 0, 1},
        {"l3v1", 1021, 0, 0, 1},
        {"l3v1", 1022, 0, 0, 1},
        {"l3v1", 1023, 0, 0, 1},
        {"l3v1", 1024, 0, 0, 1},
        {"l3v1", 1025, 0, 0, 1},
        {"l3v1", 1026, 0, 0, 1},
        {"l2v4", 1027, 0, 0, 1},
        {"l2v4", 1028, 0, 0, 1},
        {"l2v4", 1029, 0, 0, 1},
        {"l3v1", 1030, 0, 0, 1},
        {"l3v1", 1031, 0, 0, 1},
        {"l3v1", 1032, 0, 0, 1},
        {"l3v1", 1033, 0, 0, 1},
        {"l3v1", 1034, 0, 0, 1},
        {"l3v1", 1035, 0, 0, 1},
        {"l3v1", 1036, 0, 0, 1},
        {"l3v1", 1037, 0, 0, 1},
        {"l3v1", 1038, 0, 0, 1},
        {"l3v1", 1039, 0, 0, 1},
        {"l3v1", 1040, 0, 0, 1},
        {"l3v1", 1041, 0, 0, 1},
        {"l3v1", 1042, 0, 0, 1},
        {"l3v1", 1043, 0, 0, 1},
        {"l3v1", 1045, 3, 0, 1},
        {"l3v1", 1046, 3, 0, 1},
        {"l3v1", 1047, 3, 0, 1},
        {"l3v1", 1048, 3, 0, 1},
        {"l3v1", 1049, 3, 0, 1},
        {"l3v1", 1050, 3, 0, 1},
        {"l3v1", 1055, 0, 0, 1},
        {"l3v1", 1056, 0, 0, 1},
        {"l3v1", 1057, 0, 0, 1},
        {"l3v1", 1058, 0, 0, 1},
        {"l3v1", 1059, 0, 0, 1},
        {"l3v1", 1060, 0, 0, 1},
        {"l3v1", 1061, 0, 0, 1},
        {"l3v1", 1062, 0, 0, 1},
        {"l3v1", 1063, 0, 0, 1},
        {"l3v1", 1064, 0, 0, 1},
        {"l3v1", 1065, 0, 0, 1},
        {"l3v1", 1066, 0, 0, 1},
        {"l3v1", 1067, 0, 0, 1},
        {"l3v1", 1068, 0, 0, 1},
        {"l3v1", 1069, 0, 0, 1},
        {"l3v1", 1070, 0, 0, 1},
        {"l3v1", 1071, 3, 0, 1},
        {"l3v1", 1072, 3, 0, 1},
        {"l3v1", 1073, 3, 0, 1},
        {"l3v1", 1074, 3, 0, 1},
        {"l3v1", 1075, 3, 0, 1},
        {"l3v1", 1076, 3, 0, 1},
        {"l3v1", 1077, 0, 0, 1},
        {"l3v1", 1078, 0, 0, 1},
        {"l3v1", 1079, 0, 0, 1},
        {"l3v1", 1080, 0, 0, 1},
        {"l3v1", 1081, 0, 0, 1},
        {"l3v1", 1082, 0, 0, 1},
        {"l3v1", 1087, 0, 0, 1},
        {"l3v1", 1088, 0, 0, 1},
        {"l3v1", 1089, 0, 0, 1},
        {"l3v1", 1090, 0, 0, 1},
        {"l3v1", 1091, 0, 0, 1},
        {"l3v1", 1092, 0, 0, 1},
        {"l3v1", 1093, 0, 0, 1},
        {"l3v1", 1094, 3, 0, 1},
        {"l3v1", 1095, 3, 0, 1},
        {"l3v1", 1096, 0, 0, 1},
        {"l3v1", 1097, 0, 0, 1},
        {"l3v1", 1098, 0, 0, 1},
        {"l3v1", 1099, 0, 0, 1},
        {"l3v1", 1100, 0, 0, 1},
        {"l3v1", 1101, 0, 0, 1},
        {"l3v1", 1102, 0, 0, 1},
        {"l3v1", 1103, 0, 0, 1},
        {"l3v1", 1104, 0, 0, 1},
        {"l3v1", 1105, 0, 0, 1},
        {"l3v1", 1106, 3, 0, 1},
        {"l3v1", 1107, 0, 0, 1},
        {"l3v1", 1109, 0, 0, 1},
        {"l3v1", 1110, 0, 0, 1},
        {"l3v1", 1111, 0, 0, 1},
        {"l3v1", 1116, 0, 0, 1},
        {"l3v1", 1117, 0, 0, 1},
        {"l3v1", 1124, 0, 0, 1},
        {"l3v1", 1125, 0, 0, 1},
        {"l3v1", 1149, 0, 0, 1},
        {"l3v1", 1150, 0, 0, 1},
        {"l3v1", 1151, 0, 0, 1},
        {"l3v1", 1162, 0, 0, 1},
        {"l3v1", 1163, 0, 0, 1},
        {"l3v1", 1166, 0, 0, 1},
        {"l3v1", 1184, 0, 0, 1},
        {"l3v1", 1185, 0, 0, 1}
};


void getPairs(TestCase *&pairs, int& npairs) {
    pairs = testpairs;
    npairs = sizeof(testpairs) / sizeof(TestCase);

}





#if 0


/*
 * <test suite="SBML_l2v4" name="15" time="0.248"/>
<test suite="SBML_l2v4" name="16" time="0.239"/>
<test suite="SBML_l2v4" name="17" time="0.221"/>
<test suite="SBML_l2v4" name="18" time="0.246"/>
<test suite="SBML_l2v4" name="19" time="0.25"/>
<test suite="SBML_l2v4" name="20" time="0.257"/>
<test suite="SBML_l2v4" name="21" time="0.201"/>
<test suite="SBML_l2v4" name="22" time="0.243"/>
<test suite="SBML_l2v4" name="23" time="0.247"/>
<test suite="SBML_l2v4" name="24" time="0.249"/>
<test suite="SBML_l2v4" name="25" time="0.199"/>
<test suite="SBML_l2v4" name="26" time="0.258"/>
<test suite="SBML_l2v4" name="27" time="0.236"/>
<test suite="SBML_l2v4" name="28" time="0.231"/>
<test suite="SBML_l2v4" name="29" time="166.702"/>
<test suite="SBML_l2v4" name="30" time="108.012"/>
<test suite="SBML_l2v4" name="31" time="0.184"/>
<test suite="SBML_l2v4" name="889" time="0.245"/>
<test suite="SBML_l2v4" name="890" time="0.249"/>
<test suite="SBML_l2v4" name="891" time="0.229"/>
<test suite="SBML_l2v4" name="892" time="0.221"/>
<test suite="SBML_l2v4" name="893" time="0.232"/>
<test suite="SBML_l2v4" name="894" time="0.241"/>
<test suite="SBML_l2v4" name="895" time="0.262"/>
<test suite="SBML_l2v4" name="896" time="0.249"/>
 */

/**
 * The original test suite used for Valgrind testing
 */
SUITE(SBML_TEST_SUITE_VG1)
{
    {"l2v4", 15},
    {"l2v4", 16},
    {"l2v4", 17},
    {"l2v4", 18},
    {"l2v4", 19},
    {"l2v4", 20},
    {"l2v4", 21},
    {"l2v4", 22},
    {"l2v4", 23},
    {"l2v4", 24},
    {"l2v4", 25},
    {"l2v4", 26},
    {"l2v4", 27},
    {"l2v4", 29},
    {"l2v4", 30},
    {"l2v4", 31},
    {"l2v4", 889},
    {"l2v4", 890},
    {"l2v4", 891},
    {"l2v4", 892},
    {"l2v4", 893},
    {"l2v4", 894},
    {"l2v4", 895},
    {"l2v4", 896},
}

/**
 * another suite with more rate rules
 */
SUITE(SBML_TEST_SUITE_VG2)
{
    {"l2v4", 17},
    {"l2v4", 32},
    {"l2v4", 86},
    {"l2v4", 165},
    {"l2v4", 889},
    {"l3v1", 1046 },

}

#endif










