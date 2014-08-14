#ifndef rrModelGeneratorH
#define rrModelGeneratorH
#include <string>
#include <list>
#include <vector>
#include "rrExecutableModel.h"


namespace rr
{
class Compiler;

/**
 * The interface which generates executable models from sbml source.
 * This can have different concrete implementations such as compiler
 * based generators, JIT'ed or interpreter based ones.
 */
class RR_DECLSPEC ModelGenerator
{
public:
    /**
     * Factory method to create a model generator.
     * Eventauly we will have another static method that returns the names of the available
     * model generators and their descriptions. Currently we only support CModelGenerator, although
     * this is ignored and this will always return a CModelGenerator.
     *
     * The created object is owned by the caller and must be deleted accordingly.
     *
     * @param compiler: the textual name of the compiler. If this is "llvm", then then
     * the LLVM ModelGenerator is created, otherwise, A C based model generator is
     * created.
     *
     * @param tmpDir The C version stores all the generated temporary
     * C files and objects here, not used for LLVM.
     *
     * @param supportCodeDir Location where roadrunner include files are,
     * not used in LLVM.
     */
    static ModelGenerator* New(const std::string& compiler,
            const std::string& tmpDir="", const std::string& supportCodeDir="");


    enum ModelGeneratorOptions
    {
        /**
         * perform conservation analysis.
         *
         * This causes a re-ordering of the species, so results generated
         * with this flag enabled can not be compared index wise to results
         * generated otherwise.
         *
         * currently only implemented with the C code generating model
         */
        CONSERVED_MOIETIES               = (0x1 << 0),  // => 0x00000001

        /**
         * Should the model be recompiled?
         * The LLVM ModelGenerator maintins a hash table of currently running
         * models. If this flag is NOT set, then the generator will look to see
         * if there is already a running instance of the given model and
         * use the generated code from that one.
         *
         * If only a single instance of a model is run, there is no
         * need to cache the models, and this can safetly be enabled,
         * realizing some performance gains.
         */
        RECOMPILE                        = (0x1 << 1),  // => 0x00000010

        /**
         * If this is set, then a read-only model is generated. A read-only
         * model can be simulated, but no code is generated to set model
         * values, i.e. parameters, amounts, values, etc...
         *
         * It takes a finite amount of time to generate the model value setting
         * functions, and if they are not needed, one may see some performance
         * gains, especially in very large models.
         */
        READ_ONLY                        = (0x1 << 2),  // => 0x00000100

        /**
         * Generate accessor functions to allow changing of initial
         * conditions.
         */
        MUTABLE_INITIAL_CONDITIONS       = (0x1 << 3),   // => 0x00001000

        /**
         * GVN - This pass performs global value numbering and redundant load
         * elimination cotemporaneously.
         */
        OPTIMIZE_GVN                     = (0x1 << 4),

        /**
         * CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
         * simplify terminator instructions, etc...
         */
        OPTIMIZE_CFG_SIMPLIFICATION      = (0x1 << 5),

        /**
         * InstructionCombining - Combine instructions to form fewer, simple
         * instructions. This pass does not modify the CFG, and has a tendency to make
         * instructions dead, so a subsequent DCE pass is useful.
         */
        OPTIMIZE_INSTRUCTION_COMBINING   = (0x1 << 6),

        /**
         * DeadInstElimination - This pass quickly removes trivially dead instructions
         * without modifying the CFG of the function.  It is a BasicBlockPass, so it
         * runs efficiently when queued next to other BasicBlockPass's.
         */
        OPTIMIZE_DEAD_INST_ELIMINATION   = (0x1 << 7),

        /**
         * DeadCodeElimination - This pass is more powerful than DeadInstElimination,
         * because it is worklist driven that can potentially revisit instructions when
         * their other instructions become dead, to eliminate chains of dead
         * computations.
         */
        OPTIMIZE_DEAD_CODE_ELIMINATION   = (0x1 << 8),

        /**
         * InstructionSimplifier - Remove redundant instructions.
         */
        OPTIMIZE_INSTRUCTION_SIMPLIFIER  = (0x1 << 9),


        /**
         * all optimizations, use to check if bit mask has
         * any optimizations.
         */
        OPTIMIZE = OPTIMIZE_GVN | OPTIMIZE_CFG_SIMPLIFICATION |
            OPTIMIZE_INSTRUCTION_COMBINING |
            OPTIMIZE_DEAD_INST_ELIMINATION | OPTIMIZE_DEAD_CODE_ELIMINATION |
            OPTIMIZE_INSTRUCTION_SIMPLIFIER,


        /**
         * Use the LLVM MCJIT JIT engine.
         *
         * Defaults to false.
         *
         * The MCJIT is the new LLVM JIT engine, it is not as well tested as the
         * original JIT engine. Does NOT work on LLVM 3.1
         */
        USE_MCJIT =                       (0x1 << 10)
    };

    /**
     * certain model generators, such as the compiler based ones
     * generate files such as shared libraries. This specifies the
     * location where they are stored.
     */
    virtual bool setTemporaryDirectory(const std::string& path) = 0;

    /**
     * certain model generators, such as the compiler based ones
     * generate files such as shared libraries. This specifies the
     * location where they are stored.
     */
    virtual std::string getTemporaryDirectory() = 0;

    /**
     * Create an executable model from an sbml string.
     *
     * @param sbml a string containging the contents of an sbml doc.
     * @param options a bitfield containing items from the ModelGeneratorOptions
     * enumeration.
     */
    virtual ExecutableModel *createModel(const std::string& sbml,
            unsigned int options, const std::string& filename) = 0;

    /**
     * Get the compiler object that the model generator is using to
     * 'compile' sbml. Certain model generators may be interpreters, in this
     * case, the Compiler interface should still be sufficiently general to
     * manipulate interpreters as well.
     */
    virtual Compiler *getCompiler() = 0;

    /**
     * Set the name of the compiler to use. In the case of source code generating
     * model generators, this is the exectuable name of the external compiler, i.e.
     * 'gcc', 'icc', etc... For JITing generators, this may have no effect.
     */
    virtual bool setCompiler(const std::string& compiler) = 0;

    /**
     * public dtor, one can and most certainly delete an object of this class.
     */
    virtual ~ModelGenerator() {};



protected:
    ModelGenerator() {};

};
}

#endif
