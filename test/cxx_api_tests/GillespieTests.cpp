//
// Created by Ciaran on 07/05/2021.
//

#include <rrRoadRunner.h>
#include "gtest/gtest.h"

// todo move test model factor up one directory
#include "TestModelFactory.h"

#include "GillespieIntegrator.h"
#include "rrConfig.h"
using namespace rr;

/**
 * This is a more of a stub test suite
 * than a full test suite at the moment.
 * For now, it only tests the things
 * that need a new test to fix a bug.
 */

class GillespieTests : public ::testing::Test{

public:
    OpenLinearFlux openLinearFlux;
    GillespieTests() = default;
};

TEST_F(GillespieTests, Seed){
    RoadRunner rr(openLinearFlux.str());
    rr.setIntegrator("gillespie");
    rr.getIntegrator()->setValue("seed", 1234);
    ASSERT_TRUE(rr.getIntegrator()->getValue("seed") == 1234);
}

