//
// Created by Ciaran on 19/05/2021.
//

#ifndef ROADRUNNER_MOCKSOLVER_H
#define ROADRUNNER_MOCKSOLVER_H

#include "Solver.h"
#include "gmock/gmock.h"

class MockSolver : public Solver {
public:

    MOCK_METHOD(std::string, getName, (), (const, override));
        MOCK_METHOD(std::string    , getDescription       , (),         (const, override));
        MOCK_METHOD(std::string    , getHint              , (),         (const, override));
        MOCK_METHOD(void           , resetSettings        , (),         (override)) ;
        MOCK_METHOD(Setting        , getValue             , (const std::string& key),  (const));
        MOCK_METHOD(Setting        , hasValue             , (const std::string& key),  (const));
        MOCK_METHOD(size_t         , getNumParams         , (),                        (const));
        MOCK_METHOD(std::string    , getParamName         , (size_t n),  (const));
        MOCK_METHOD(std::string    , getParamDisplayName  , (int n),     (const));
        MOCK_METHOD(std::string    , getParamHint         , (int n),     (const));
        MOCK_METHOD(std::string    , getParamDesc         , (int n),     (const));
        MOCK_METHOD(std::string    , getValueAsString     , (const std::string& key),   ());
        MOCK_METHOD(void           , setValue             , (const std::string& key, Setting value), ());
        MOCK_METHOD(std::string    , getSettingsRepr      , (),  (const));
        MOCK_METHOD(std::string    , toString             , (),  (const));
        MOCK_METHOD(std::string    , toRepr               , (),  (const));
        MOCK_METHOD(ExecutableModel, *getModel            , (),  (const));

};

#endif //ROADRUNNER_MOCKSOLVER_H
