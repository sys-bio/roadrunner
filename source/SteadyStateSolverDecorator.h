//
// Created by Ciaran on 12/03/2021.
//

#ifndef ROADRUNNER_STEADYSTATESOLVERDECORATOR_H
#define ROADRUNNER_STEADYSTATESOLVERDECORATOR_H

#include "SteadyStateSolver.h"
#include <iostream>

namespace rr {

    /**
     * The base class Decorator follows the same interface as other
     * SteadyStateSolvers. The primary purpose of this class is the define
     * the wrapping interface for all concrete decorators. The default
     * implementation of the wrapping code might include a field for
     * storing a wrapped component and a means to initialize it.
     */
    class SteadyStateSolverDecorator : public SteadyStateSolver {
    public:
        using SteadyStateSolver::SteadyStateSolver;

        ~SteadyStateSolverDecorator() override = default;

        SteadyStateSolverDecorator() = default;

        explicit SteadyStateSolverDecorator(SteadyStateSolver *solver);

        /**
         * Base decorator delegates all work to the wrapped steady state solver.
         * Subclass decorators may modify any or all of these operations.
         */

        void syncWithModel(ExecutableModel *m) override;

        double solve() override;

        std::string getName() const override;

        std::string getDescription() const override;

        std::string getHint() const override;

        void resetSettings() override;

    protected:
        SteadyStateSolver *solver_;

    private:

        /**
         * @brief get the name of this decorator.
         * @details used for modifying the name, description
         * and hint for solver
         */
        virtual std::string decoratorName() const;

    };



//
///**
// * Concrete decorators call the wrapped object and alter its
// * result in some way.
// */
//    class ConcreteDecoratorA : public Decorator {
//    public:
//        ConcreteDecoratorA(Component *component)
//                : Decorator(component) {}
//
//        std::string Operation() override {
//            return "ConcreteDecoratorA(" + Decorator::Operation() + ")";
//        }
//    };
//
//    class ConcreteDecoratorB : public Decorator {
//    public:
//        ConcreteDecoratorB(Component *component)
//                : Decorator(component) {}
//
//        std::string Operation() override {
//            return "ConcreteDecoratorB(" + Decorator::Operation() + ")";
//        }
//    };
//
///**
// * The client code works with all objects using the Component interface.
// * This way it remains independent of the concrete classes of components
// * it works with
// */
//    void clientCode(Component *component) {
//        std::cout << component->Operation() << std::endl;
//    }
}

#endif //ROADRUNNER_STEADYSTATESOLVERDECORATOR_H
