/*
 * ConservedMoietyPlugin.h
 *
 *  Created on: Oct 18, 2013
 *      Author: andy
 */

#ifndef ConservedMoietyPlugin_H_
#define ConservedMoietyPlugin_H_

#include "rrExporter.h"
#include "ConservationExtension.h"
#include <sbml/extension/SBasePlugin.h>

namespace rr
{
namespace conservation
{

class RR_DECLSPEC ConservedMoietyPlugin: public libsbml::SBasePlugin
{
public:

    /**
     * Constructor
     */
    ConservedMoietyPlugin(const std::string &uri, const std::string &prefix,
            ConservationPkgNamespaces *consns);

    /**
     * Copy constructor. Creates a copy of this SBase object.
     */
    ConservedMoietyPlugin(const ConservedMoietyPlugin& orig);

    /**
     * Destroy this object.
     */
    virtual ~ConservedMoietyPlugin();

    /**
     * Assignment operator for ConservedMoietyPlugin.
     */
    ConservedMoietyPlugin& operator=(const ConservedMoietyPlugin& orig);

    /**
     * Creates and returns a deep copy of this ConservedMoietyPlugin object.
     *
     * @return a (deep) copy of this SBase object
     */
    virtual ConservedMoietyPlugin* clone() const;

    bool getConservedMoiety() const;

    void setConservedMoiety(bool value);

    // gets the global parameter associated with this
    // species if it forms part of a conserved cycle
    std::string getConservedQuantity() const;

    const std::vector<std::string>& getConservedQuantities() const;

    void setConservedQuantity(const std::string& id);

    void addConservedQuantity(const std::string& id);

    /**
     * Subclasses must override this method to read values from the given
     * XMLAttributes if they have their specific attributes.
     */
    virtual void readAttributes (const libsbml::XMLAttributes& attributes,
                                 const libsbml::ExpectedAttributes& expectedAttributes);


    /**
     * Subclasses must override this method to write their XML attributes
     * to the XMLOutputStream if they have their specific attributes.
     */
    virtual void writeAttributes (libsbml::XMLOutputStream& stream) const;

    virtual libsbml::List* getAllElements(libsbml::ElementFilter* filter=NULL)
    {
        return NULL;
    }

private:
    bool conservedMoiety;
    std::vector<std::string> conservedQuantities;

};

}
} // namespace rr } namespace conservation }
#endif /* ConservedMoietyPlugin_H_ */
