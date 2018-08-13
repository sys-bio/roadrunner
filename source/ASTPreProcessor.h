#ifndef ASTPreProcessorH
#define ASTPreProcessorH

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

	typedef std::map<std::string, unsigned int> valMap;

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