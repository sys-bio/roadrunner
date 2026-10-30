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
        std::ostringstream os;
        os << "No such solver called \""<< name << "\". ";
        os << "Available options are ";
        for (auto& it : registrars){
            os << "\"" << it->getName() << "\", ";
        }
        std::string errMsg = os.str();
        // remove the extra space and ","
        errMsg.erase(errMsg.end()-1);
        errMsg.erase(errMsg.end()-1);
        throw InvalidKeyException(errMsg);
    }

    void RegistrationFactory::registerSolver(Registrable* i) {
        if (!i)
            throw CoreException("Registrar is null");
        // Several copies of this code in one process (roadrunner and rrplugins, say) can share
        // one factory, so a solver that is already registered is not added a second time.
        for (auto existing : registrars) {
            if (existing->getName() == i->getName()) {
                delete i;
                return;
            }
        }
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

