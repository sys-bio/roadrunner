// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

// == FILEDOC =================================================

/** @file GPUSimModel.hpp
  * @author JKM
  * @date 09/08/2014
  * @copyright Apache License, Version 2.0
  * @brief General ODE models
**/

#ifndef rrGPUSimModelH
#define rrGPUSimModelH

// == MACROS ==================================================

#if !defined(__cplusplus)
#   error "You are including a .hpp in non-C++ code"
#endif

#if (__cplusplus < 201103L) && !defined(_MSC_VER)
#   error "This file requires C++11 support in order to compile"
#endif

// == INCLUDES ================================================

#include "rrSelectionRecord.h"
#include "GPUSimException.h"
#include "GPUSimReal.h"
#include "patterns/AccessPtrIterator.hpp"
#include "patterns/Range.hpp"

#include <memory>
#include <set>

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

/**
 * @brief Base class of model elements
 */
class RR_DECLSPEC ModelElement
{
public:
    virtual ~ModelElement();

    virtual bool matchesType(int typebits) const { return false; }

    virtual std::string getId() const = 0;
};

/**
 * @brief Math AST node
 */
class RR_DECLSPEC ModelASTNode
{
public:
    virtual ~ModelASTNode();
};
typedef std::unique_ptr<ModelASTNode> ModelASTNodePtr;

/**
 * @brief Binary AST node
 */
class RR_DECLSPEC BinaryASTNode : public ModelASTNode
{
public:
    // Construct from left / right operands
    BinaryASTNode(ModelASTNodePtr&& left, ModelASTNodePtr&& right)
      : left_(std::move(left)), right_(std::move(right)) {}

    ModelASTNode* getLeft() { return left_.get(); }
    const ModelASTNode* getLeft() const { return left_.get(); }

    ModelASTNode* getRight() { return right_.get(); }
    const ModelASTNode* getRight() const { return right_.get(); }

protected:
    ModelASTNodePtr left_;
    ModelASTNodePtr right_;
};

/**
 * @brief Sum AST node
 */
class RR_DECLSPEC SumASTNode : public BinaryASTNode
{
public:
    using BinaryASTNode::BinaryASTNode;
};

/**
 * @brief Product AST node
 */
class RR_DECLSPEC ProductASTNode : public BinaryASTNode
{
public:
    using BinaryASTNode::BinaryASTNode;
};

/**
 * @brief Quotient AST node
 */
class RR_DECLSPEC QuotientASTNode : public BinaryASTNode
{
public:
    using BinaryASTNode::BinaryASTNode;
};

/**
 * @brief Binary AST node
 */
class RR_DECLSPEC ExponentiationASTNode : public BinaryASTNode
{
public:
    using BinaryASTNode::BinaryASTNode;
};

/**
 * @brief Literal AST node
 */
class RR_DECLSPEC LiteralASTNode : public ModelASTNode
{
public:

protected:
};

/**
 * @brief Integer literal AST node
 */
class RR_DECLSPEC IntegerLiteralASTNode : public LiteralASTNode
{
public:
    /// Construct from value
    IntegerLiteralASTNode(long val)
      : val_(val) {}

    /// Get the stored value
    long getValue() const { return val_; }

    /// Set the stored value
    void setValue(long v) { val_ = v; }

protected:
    long val_;
};

/**
 * @brief Container for AST-based algebra
 */
class RR_DECLSPEC ModelAlgebra
{
public:

    ModelASTNode* getRoot() { return root_.get(); }
    const ModelASTNode* getRoot() const { return root_.get(); }

    void setRoot(ModelASTNodePtr&& root) {
        root_ = std::move(root);
    }

protected:
    ModelASTNodePtr root_;
};

} // namespace rrgpu

} // namespace rr

#endif /* GPUSimModelH */
