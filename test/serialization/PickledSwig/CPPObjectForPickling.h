//
// Created by Ciaran on 22/04/2021.
//

#ifndef ROADRUNNER_CPPOBJECTFORPICKLING_H
#define ROADRUNNER_CPPOBJECTFORPICKLING_H


/**
 * Note: This source file is not used in roadrunner testing
 * but its an example of how to support pickled swig objects.
 */

#include <iostream>

class CPPObjectForPickling{
public:

    CPPObjectForPickling() = default;

    int getInt() const;

    void setInt(int int_);

    const std::string &getString() const;

    void setString(const std::string &string_);

    double getDouble() const;

    void setDouble(double double_);

    int times2();

private:
    int int_ = 0;
    std::string string_;
    double double_ = 0.0;

};


#endif //ROADRUNNER_CPPOBJECTFORPICKLING_H
