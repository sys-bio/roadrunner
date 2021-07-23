//
// Created by Ciaran Welsh on 14/06/2021.
//

#include "RegistrationFactory.h"
#include "rrException.h"
#include "Solver.h"

#include <mutex>

namespace rr {
    static std::mutex mutex;

    RegistrationFactory::~RegistrationFactory() {
        for (auto it : registrars){
            delete it;
        }
    }

    Solver* RegistrationFactory::New(const std::string& name, ExecutableModel* m) const {
        for (auto it : registrars){
            if (it->getName() == name) {
                return it->construct(m);
            }
        }
        throw InvalidKeyException("No such integrator: " + name);
    }

    void RegistrationFactory::registerSolver(Registrable* i) {
        if (!i)
            throw CoreException("Registrar is null");
        registrars.push_back(i);
    }

    std::size_t RegistrationFactory::size() const {
        return registrars.size();
    }

    std::string RegistrationFactory::name(std::size_t n) const {
        return registrars.at(n)->getName();
    }

    std::string RegistrationFactory::hint(std::size_t n) const {
        return registrars.at(n)->getHint();
    }

    std::string RegistrationFactory::description(std::size_t n) const {
        return registrars.at(n)->getDescription();
    }


}

