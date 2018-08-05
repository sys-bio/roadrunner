#ifndef ASTPreProcessorH
#define ASTPreProcessorH

#define valMap std::map<std::string, unsigned int>

#include <map>

namespace libsbml
{
class ASTNode;
}

namespace rr
{
class ASTPreProcessor
{
  public:
    ASTPreProcessor();
    ~ASTPreProcessor();

    libsbml::ASTNode *preProcess(libsbml::ASTNode *ast, valMap *values);

  private:
    /**
     * sbml ASTNode does not contain as assigment '=' type, assigment
     * is handled by other sbml elements such as initialAssigment.
     *
     */
    libsbml::ASTNode *applyArithmetic(libsbml::ASTNode *ast, valMap *values = NULL);

    libsbml::ASTNode *selector(libsbml::ASTNode *parent, libsbml::ASTNode *ast, valMap *values = NULL);
};
}
#endif