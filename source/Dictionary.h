/*
 * Dictionary.h
 *
 *  Created on: Jul 14, 2014
 *      Author: andy
 */

#ifndef DICTIONARY_H_
#define DICTIONARY_H_

#include "Variant.h"
#include <string>
#include <vector>


namespace rr {

class Dictionary
{
public:
    /**
     * set an arbitrary key
     */
    virtual void setValue(const std::string& key, const rr::Variant& value) = 0;

    virtual Variant getValue(const std::string& key) const = 0;

    virtual bool hasKey(const std::string& key) const = 0;

    virtual int deleteValue(const std::string& key) = 0;

    virtual std::vector<std::string> getKeys() const = 0;

    virtual ~Dictionary() {};
};

}





#endif /* DICTIONARY_H_ */
