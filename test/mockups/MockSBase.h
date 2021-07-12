//
// Created by Ciaran Welsh on 27/05/2021.
//

#ifndef ROADRUNNER_MOCKSBASE_H
#define ROADRUNNER_MOCKSBASE_H

#include "gmock/gmock.h"
#include "sbml/SBase.h"

using namespace libsbml;

/**
 * A mockup of libsbml::SBase for use in roadrunner testing.
 *
 * This was a nice idea, but this can't actually be used unless
 * roadrunner has classes that depend on the libsbml::SBase abstraction
 * rather than libsbml::Model or libsbml::Doc.
 *
 * Kept only to prevent need for rewriting in future, should the
 * occasion call for its use.
 */
class MockSBase : public libsbml::SBase {
public:
    MOCK_METHOD(bool, accept, (SBMLVisitor & v), (const, override));
    MOCK_METHOD(SBase*, clone, (), (const, override));
    MOCK_METHOD(SBase*, getElementBySId, (const std::string& id), (override));
    MOCK_METHOD(SBase*, getElementByMetaId, (const std::string& metaid), (override));
    MOCK_METHOD(List*, getAllElements, (ElementFilter * filter), (override));
    MOCK_METHOD(void, renameSIdRefs, (const std::string& oldid, const std::string& newid), (override));
    MOCK_METHOD(void, renameMetaIdRefs, (const std::string& oldid, const std::string& newid), (override));
    MOCK_METHOD(void, renameUnitSIdRefs, (const std::string& oldid, const std::string& newid), (override));
    MOCK_METHOD(void, replaceSIDWithFunction, (const std::string& id, const ASTNode* function), (override));
    MOCK_METHOD(void, divideAssignmentsToSIdByFunction, (const std::string& id, const ASTNode* function), (override));
    MOCK_METHOD(void, multiplyAssignmentsToSIdByFunction, (const std::string& id, const ASTNode* function), (override));
    MOCK_METHOD(SBase*, getElementFromPluginsBySId, (std::string id), (override));
    MOCK_METHOD(SBase*, getElementFromPluginsByMetaId, (std::string metaid), (override));
    MOCK_METHOD(bool, hasNonstandardIdentifierBeginningWith, (const std::string& prefix), (override));
    MOCK_METHOD(int, prependStringToAllIdentifiers, (const std::string& prefix), (override));
    MOCK_METHOD(int, transformIdentifiers, (IdentifierTransformer * idTransformer), (override));
    MOCK_METHOD(List*, getAllElementsFromPlugins, (ElementFilter * filter), (override));
    MOCK_METHOD(const std::string&, getId, (), (const, override));
    MOCK_METHOD(const std::string&, getName, (), (const, override));
    MOCK_METHOD(XMLNamespaces*, getNamespaces, (), (const, override));
    MOCK_METHOD(bool, isSetId, (), (const));
    MOCK_METHOD(int, setId, (const std::string& sid), (override));
    MOCK_METHOD(int, setIdAttribute, (const std::string& sid), (override));
    MOCK_METHOD(int, setName, (const std::string& name), (override));
    MOCK_METHOD(int, setAnnotation, (const XMLNode* annotation), (override));
    MOCK_METHOD(int, setAnnotation, (const std::string& annotation), (override));
    MOCK_METHOD(int, appendAnnotation, (const XMLNode* annotation), (override));
    MOCK_METHOD(int, appendAnnotation, (const std::string& annotation), (override));
    MOCK_METHOD(void, setSBMLDocument, (SBMLDocument * d), (override));
    MOCK_METHOD(void, connectToParent, (SBase * parent), (override));
    MOCK_METHOD(void, connectToChild, (), (override));
    MOCK_METHOD(int, setSBOTerm, (int value), (override));
    MOCK_METHOD(int, setSBOTerm, (const std::string &sboid), (override));
    MOCK_METHOD(int, unsetId, (), (override));
    MOCK_METHOD(int, unsetName, (), (override));
    MOCK_METHOD(int, getTypeCode, (), (const, override));
    MOCK_METHOD(const std::string&, getElementName, (), (const, override));
    MOCK_METHOD(int, getAttribute, (const std::string& attributeName, double& value), (const, override));
    MOCK_METHOD(int, getAttribute, (const std::string& attributeName, bool& value), (const, override));
    MOCK_METHOD(int, getAttribute, (const std::string& attributeName, int& value), (const, override));
    MOCK_METHOD(int, getAttribute, (const std::string& attributeName, unsigned int& value), ( const, override));
    MOCK_METHOD(int, getAttribute, (const std::string& attributeName, std::string& value), (const, override));
    MOCK_METHOD(bool, isSetAttribute, (const std::string& attributeName), (const, override));
    MOCK_METHOD(int, setAttribute, (const std::string& attributeName, double value), (override));
    MOCK_METHOD(int, setAttribute, (const std::string& attributeName, bool value), (override));
    MOCK_METHOD(int, setAttribute, (const std::string& attributeName, int value), (override));
    MOCK_METHOD(int, setAttribute, (const std::string& attributeName, unsigned int value), (override));
    MOCK_METHOD(int, setAttribute, (const std::string& attributeName, const std::string& value), (override));
    MOCK_METHOD(int, unsetAttribute, (const std::string& attributeName), (override));
    MOCK_METHOD(SBase*, createChildObject, (const std::string& elementName), (override));
    MOCK_METHOD(int, addChildObject, (const std::string& elementName, const SBase* element), (override));
    MOCK_METHOD(SBase*, removeChildObject, (const std::string& elementName, const std::string& id), (override));
    MOCK_METHOD(unsigned int, getNumObjects, (const std::string& objectName), (override));
    MOCK_METHOD(SBase*, getObject, (const std::string& objectName, unsigned int index), (override));
    MOCK_METHOD(int, setMessage, (const std::string& message, bool addXHTMLMarkup), (override));
    MOCK_METHOD(const ASTNode*, getMath, (), (const, override));
    MOCK_METHOD(bool, isSetMath, (), (const, override));
    MOCK_METHOD(int, setMath, (const ASTNode* math), (override));
    MOCK_METHOD(std::string, getMessageString, (), (const, override));
    MOCK_METHOD(void, enablePackageInternal, (const std::string& pkgURI, const std::string& pkgPrefix, bool flag),(override));
    MOCK_METHOD(void, writeExtensionElements, (XMLOutputStream & stream), (const, override));
    MOCK_METHOD(void, write, (XMLOutputStream & stream), (const, override));
    MOCK_METHOD(void, writeElements, (XMLOutputStream & stream), (const, override));
    MOCK_METHOD(bool, hasRequiredAttributes, (), (const, override));
    MOCK_METHOD(bool, hasRequiredElements, (), (const, override));
    MOCK_METHOD(SBMLNamespaces *, getSBMLNamespaces, (), (const, override));
    MOCK_METHOD(UnitDefinition*, getDerivedUnitDefinition, (), (override));
    MOCK_METHOD(bool, containsUndeclaredUnits, (), (override));
    MOCK_METHOD(int, removeFromParentAndDelete, (), (override));
    MOCK_METHOD(bool, hasOptionalAttributes, (), (const, override));
    MOCK_METHOD(bool, hasOptionalElements, (), (const, override));
    MOCK_METHOD(void, updateSBMLNamespace, (const std::string& package, unsigned int level, unsigned int version),(override));
    MOCK_METHOD(void, setElementText, (const std::string &text), (override));
    MOCK_METHOD(SBase*, createObject, (XMLInputStream & stream), (override));
    MOCK_METHOD(bool, readOtherXML, (XMLInputStream & stream), (override));
    MOCK_METHOD(int, getElementPosition, (), (const, override));
    MOCK_METHOD(void, addExpectedAttributes, (ExpectedAttributes & attributes), (override));
    MOCK_METHOD(void, readAttributes, (const XMLAttributes& attributes, const ExpectedAttributes& expectedAttributes),(override));
    MOCK_METHOD(void, writeAttributes, (XMLOutputStream & stream), (const, override));
    MOCK_METHOD(void, writeXMLNS, (XMLOutputStream & stream), (const, override));
    MOCK_METHOD(void, syncAnnotation, (), (override));
    MOCK_METHOD(SBase*, createExtensionObject, (XMLInputStream & stream), (override));
    MOCK_METHOD(void, readExtensionAttributes,(const XMLAttributes& attributes, const ExpectedAttributes* expectedAttributes), (override));
    MOCK_METHOD(void, writeExtensionAttributes, (XMLOutputStream & stream), (const, override));
};

#endif //ROADRUNNER_MOCKSBASE_H
