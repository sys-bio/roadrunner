//
// Created by Ciaran on 22/04/2021.
//

#ifndef ROADRUNNER_IPYPICKLE_H
#define ROADRUNNER_IPYPICKLE_H

namespace rr {

/**
 * @brief Interface for objects that need to be pickled
 * at the Python level
 * @details Classes that implement this interface have the required
 * information to enquire which attributes are necessary
 * for serialization / deserialisation using Pickle
 */
    class IPyPickle {

        /**
         * @brief default constructor
         */
        IPyPickle() = default;

        /**
         * @brief Returns a list of attributes that define
         * this object.
         * @details For instance a class with a string, int and double
         * member variables would be defined by those three values, since
         * another equivalent object could be constructed using these three
         * data
         */
        std::vector <std::string> definingAttributes() = 0;

    };
}
#endif //ROADRUNNER_IPYPICKLE_H
