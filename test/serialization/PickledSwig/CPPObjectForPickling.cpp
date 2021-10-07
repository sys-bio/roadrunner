//
// Created by Ciaran on 22/04/2021.
//

#include "CPPObjectForPickling.h"

int CPPObjectForPickling::getInt() const {
    return int_;
}

void CPPObjectForPickling::setInt(int int_) {
    CPPObjectForPickling::int_ = int_;
}

const std::string &CPPObjectForPickling::getString() const {
    return string_;
}

void CPPObjectForPickling::setString(const std::string &string) {
    string_ = string;
}

double CPPObjectForPickling::getDouble() const {
    return double_;
}

void CPPObjectForPickling::setDouble(double double_) {
    CPPObjectForPickling::double_ = double_;
}

int CPPObjectForPickling::times2() {
    return int_*2;
}
