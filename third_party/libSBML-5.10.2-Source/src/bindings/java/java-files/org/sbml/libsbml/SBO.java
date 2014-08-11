/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 2.0.4
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package org.sbml.libsbml;

/** 
 *  Facilities for using the Systems Biology Ontology.
 <p>
 * <p style='color: #777; font-style: italic'>
This class of objects is defined by libSBML only and has no direct
equivalent in terms of SBML components.  This class is not prescribed by
the SBML specifications, although it is used to implement features
defined in SBML.
</p>

 <p>
 * The values of 'id' attributes on SBML components allow the components to
 * be cross-referenced within a model. The values of 'name' attributes on
 * SBML components provide the opportunity to assign them meaningful labels
 * suitable for display to humans.  The specific identifiers and labels
 * used in a model necessarily must be unrestricted by SBML, so that
 * software and users are free to pick whatever they need.  However, this
 * freedom makes it more difficult for software tools to determine, without
 * additional human intervention, the semantics of models more precisely
 * than the semantics provided by the SBML object classes defined in other
 * sections of this document.  For example, there is nothing inherent in a
 * parameter with identifier <code>k</code> that would indicate to a
 * software tool it is a first-order rate constant (if that's what
 * <code>k</code> happened to be in some given model).  However, one may
 * need to convert a model between different representations (e.g.,
 * Henri-Michaelis-Menten versus elementary steps), or to use it with
 * different modeling approaches (discrete or continuous).  One may also
 * need to relate the model components with other description formats such
 * as SBGN (<a target='_blank'
 * href='http://www.sbgn.org/'>http://www.sbgn.org/</a>) using deeper
 * semantics.  Although an advanced software tool <em>might</em> be able to
 * deduce the semantics of some model components through detailed analysis
 * of the kinetic rate expressions and other parts of the model, this
 * quickly becomes infeasible for any but the simplest of models.
 <p>
 * An approach to solving this problem is to associate model components
 * with terms from carefully curated controlled vocabularies (CVs).  This
 * is the purpose of the optional 'sboTerm' attribute provided on the SBML
 * class {@link SBase}.  The 'sboTerm' attribute always refers to terms belonging
 * to the Systems Biology Ontology (SBO).
 <p>
 * <h2>Use of {@link SBO}</h2>
 <p>
 * Labeling model components with terms from shared controlled vocabularies
 * allows a software tool to identify each component using identifiers that
 * are not tool-specific.  An example of where this is useful is the desire
 * by many software developers to provide users with meaningful names for
 * reaction rate equations.  Software tools with editing interfaces
 * frequently provide these names in menus or lists of choices for users.
 * However, without a standardized set of names or identifiers shared
 * between developers, a given software package cannot reliably interpret
 * the names or identifiers of reactions used in models written by other
 * tools.
 <p>
 * The first solution that might come to mind is to stipulate that certain
 * common reactions always have the same name (e.g., 'Michaelis-Menten'), but
 * this is simply impossible to do: not only do humans often disagree on
 * the names themselves, but it would not allow for correction of errors or
 * updates to the list of predefined names except by issuing new releases
 * of the SBML specification&mdash;to say nothing of many other limitations
 * with this approach.  Moreover, the parameters and variables that appear
 * in rate expressions also need to be identified in a way that software
 * tools can interpret mechanically, implying that the names of these
 * entities would also need to be regulated.
 <p>
 * The Systems Biology Ontology (SBO) provides terms for identifying most
 * elements of SBML. The relationship implied by an 'sboTerm' on an SBML
 * model component is <em>is-a</em> between the characteristic of the
 * component meant to be described by SBO on this element and the SBO
 * term identified by the value of the 'sboTerm'. By adding SBO term
 * references on the components of a model, a software tool can provide
 * additional details using independent, shared vocabularies that can
 * enable <em>other</em> software tools to recognize precisely what the
 * component is meant to be.  Those tools can then act on that information.
 * For example, if the SBO identifier <code>'SBO:0000049'</code> is assigned
 * to the concept of 'first-order irreversible mass-action kinetics,
 * continuous framework', and a given {@link KineticLaw} object in a model has an
 * 'sboTerm' attribute with this value, then regardless of the identifier
 * and name given to the reaction itself, a software tool could use this to
 * inform users that the reaction is a first-order irreversible mass-action
 * reaction.  This kind of reverse engineering of the meaning of reactions
 * in a model would be difficult to do otherwise, especially for more
 * complex reaction types.
 <p>
 * The presence of SBO labels on {@link Compartment}, {@link Species}, and {@link Reaction}
 * objects in SBML can help map those entities to equivalent concepts in
 * other standards, such as (but not limited to) BioPAX (<a target='_blank'
 * href='http://www.biopax.org/'>http://www.biopax.org/</a>), PSI-MI (<a
 * target='_blank'
 * href='http://www.psidev.info/index.php?q=node/60'>http://www.psidev.info</a>),
 * or the Systems Biology Graphical Notation (SBGN, <a target='_blank'
 * href='http://www.sbgn.org/'>http://www.sbgn.org/</a>).  Such mappings
 * can be used in conversion procedures, or to build interfaces, with SBO
 * becoming a kind of 'glue' between standards of representation.
 <p>
 * The presence of the label on a kinetic expression can also allow
 * software tools to make more intelligent decisions about reaction rate
 * expressions.  For example, an application could recognize certain types
 * of reaction formulas as being ones it knows how to solve with optimized
 * procedures.  The application could then use internal, optimized code
 * implementing the rate formula indexed by identifiers such as
 * <code>'SBO:0000049'</code> appearing in SBML models.
 <p>
 * Finally, SBO labels may be very valuable when it comes to model
 * integration, by helping identify interfaces, convert mathematical
 * expressions and parameters etc.
 <p>
 * Although the use of SBO can be beneficial, it is critical to keep in
 * mind that the presence of an 'sboTerm' value on an object <em>must not
 * change the fundamental mathematical meaning</em> of the model.  An SBML
 * model must be defined such that it stands on its own and does not depend
 * on additional information added by SBO terms for a correct mathematical
 * interpretation.  SBO term definitions will not imply any alternative
 * mathematical semantics for any SBML object labeled with that term.  Two
 * important reasons motivate this principle.  First, it would be too
 * limiting to require all software tools to be able to understand the SBO
 * vocabularies in addition to understanding SBML.  Supporting SBO is not
 * only additional work for the software developer; for some kinds of
 * applications, it may not make sense.  If SBO terms on a model are
 * optional, it follows that the SBML model <em>must</em> remain
 * unambiguous and fully interpretable without them, because an application
 * reading the model may ignore the terms.  Second, we believe allowing the
 * use of 'sboTerm' to alter the mathematical meaning of a model would
 * allow too much leeway to shoehorn inconsistent concepts into SBML
 * objects, ultimately reducing the interoperability of the models.
 <p>
 * <h2>Relationships between {@link SBO} and SBML</h2>
 <p>
 * The goal of SBO labeling for SBML is to clarify to the fullest extent
 * possible the nature of each element in a model.  The approach taken in
 * SBO begins with a hierarchically-structured set of controlled
 * vocabularies with six main divisions: (1) entity, (2) participant role,
 * (3) quantitative parameter, (4) modeling framework, (5) mathematical
 * expression, and (6) interaction.  The web site for SBO (<a
 * target='_blank'
 * href='http://biomodels.net/sbo'>http://biomodels.net</a>) should be
 * consulted for the current version of the ontology.
 <p>
 * The Systems Biology Ontology (SBO) is not part of SBML; it is being
 * developed separately, to allow the modeling community to evolve the
 * ontology independently of SBML.  However, the terms in the ontology are
 * being designed keeping SBML components in mind, and are classified into
 * subsets that can be directly related with SBML components such as
 * reaction rate expressions, parameters, and others.  The use of 'sboTerm'
 * attributes is optional, and the presence of 'sboTerm' on an element does
 * not change the way the model is <em>interpreted</em>.  Annotating SBML
 * elements with SBO terms adds additional semantic information that may
 * be used to <em>convert</em> the model into another model, or another
 * format.  Although SBO support provides an important source of
 * information to understand the meaning of a model, software does not need
 * to support 'sboTerm' to be considered SBML-compliant.
 */

public class SBO {
   private long swigCPtr;
   protected boolean swigCMemOwn;

   protected SBO(long cPtr, boolean cMemoryOwn)
   {
     swigCMemOwn = cMemoryOwn;
     swigCPtr    = cPtr;
   }

   protected static long getCPtr(SBO obj)
   {
     return (obj == null) ? 0 : obj.swigCPtr;
   }

   protected static long getCPtrAndDisown (SBO obj)
   {
     long ptr = 0;

     if (obj != null)
     {
       ptr             = obj.swigCPtr;
       obj.swigCMemOwn = false;
     }

     return ptr;
   }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        libsbmlJNI.delete_SBO(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'quantiative parameter'</em>, <code>false</code>
   * otherwise.
   <p>
   * 
   */ public
 static boolean isQuantitativeParameter(long term) {
    return libsbmlJNI.SBO_isQuantitativeParameter(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'participant role'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isParticipantRole(long term) {
    return libsbmlJNI.SBO_isParticipantRole(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'modeling framework'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isModellingFramework(long term) {
    return libsbmlJNI.SBO_isModellingFramework(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'mathematical expression'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isMathematicalExpression(long term) {
    return libsbmlJNI.SBO_isMathematicalExpression(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'kinetic constant'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isKineticConstant(long term) {
    return libsbmlJNI.SBO_isKineticConstant(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'reactant'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isReactant(long term) {
    return libsbmlJNI.SBO_isReactant(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'product'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isProduct(long term) {
    return libsbmlJNI.SBO_isProduct(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'modifier'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isModifier(long term) {
    return libsbmlJNI.SBO_isModifier(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'rate law'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isRateLaw(long term) {
    return libsbmlJNI.SBO_isRateLaw(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'event'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isEvent(long term) {
    return libsbmlJNI.SBO_isEvent(term);
  }

  
/**
    * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
    <p>
    * @return <code>true</code> if <code>term</code> is-a SBO <em>'physical participant</em>, <code>false</code> otherwise.
   <p>
   * 
    */ public
 static boolean isPhysicalParticipant(long term) {
    return libsbmlJNI.SBO_isPhysicalParticipant(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'participant'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isParticipant(long term) {
    return libsbmlJNI.SBO_isParticipant(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'interaction'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isInteraction(long term) {
    return libsbmlJNI.SBO_isInteraction(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'entity'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isEntity(long term) {
    return libsbmlJNI.SBO_isEntity(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'functional entity'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isFunctionalEntity(long term) {
    return libsbmlJNI.SBO_isFunctionalEntity(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'material entity'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isMaterialEntity(long term) {
    return libsbmlJNI.SBO_isMaterialEntity(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'conservation law'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isConservationLaw(long term) {
    return libsbmlJNI.SBO_isConservationLaw(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'steady state expression'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isSteadyStateExpression(long term) {
    return libsbmlJNI.SBO_isSteadyStateExpression(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'functional compartment'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isFunctionalCompartment(long term) {
    return libsbmlJNI.SBO_isFunctionalCompartment(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'continuous framework'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isContinuousFramework(long term) {
    return libsbmlJNI.SBO_isContinuousFramework(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'discrete framework'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isDiscreteFramework(long term) {
    return libsbmlJNI.SBO_isDiscreteFramework(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'logical framework'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isLogicalFramework(long term) {
    return libsbmlJNI.SBO_isLogicalFramework(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'metadata representation'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isMetadataRepresentation(long term) {
    return libsbmlJNI.SBO_isMetadataRepresentation(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'occurring entity representation'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isOccurringEntityRepresentation(long term) {
    return libsbmlJNI.SBO_isOccurringEntityRepresentation(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'physical entity representation'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isPhysicalEntityRepresentation(long term) {
    return libsbmlJNI.SBO_isPhysicalEntityRepresentation(term);
  }

  
/**
   * Returns <code>true</code> if the given term identifier comes from the stated branch of SBO.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'systems description parameter'</em>, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isSystemsDescriptionParameter(long term) {
    return libsbmlJNI.SBO_isSystemsDescriptionParameter(term);
  }

  
/**
   * Predicate for checking whether the given term is obsolete.
   <p>
   * @return <code>true</code> if <code>term</code> is-a SBO <em>'obsolete'</em> term, <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean isObselete(long term) {
    return libsbmlJNI.SBO_isObselete(term);
  }

  
/**
   * Returns the integer as a correctly formatted SBO identifier string.
   <p>
   * @return the given integer sboTerm as a zero-padded seven digit string.
   <p>
   * @note If the sboTerm is not in the correct range
   * (0000000&ndash;9999999), an empty string is returned.
   <p>
   * 
   */ public
 static String intToString(int sboTerm) {
    return libsbmlJNI.SBO_intToString(sboTerm);
  }

  
/**
   * Returns the string as a correctly formatted SBO integer portion.
   <p>
   * @return the given string sboTerm as an integer.  If the sboTerm is not
   * in the correct format (a zero-padded, seven digit string), <code>-1</code> is
   * returned.
   <p>
   * 
   */ public
 static int stringToInt(String sboTerm) {
    return libsbmlJNI.SBO_stringToInt(sboTerm);
  }

  
/**
   * Checks the format of the given SBO identifier string.
   <p>
   * @return <code>true</code> if sboTerm is in the correct format (a zero-padded, seven
   * digit string), <code>false</code> otherwise.
   <p>
   * 
   */ public
 static boolean checkTerm(String sboTerm) {
    return libsbmlJNI.SBO_checkTerm__SWIG_0(sboTerm);
  }

  
/**
   * Checks the format of the given SBO identifier, given in the form of
   * the integer portion alone.
   <p>
   * @return <code>true</code> if sboTerm is in the range (0000000&ndash;9999999), <code>false</code>
   * otherwise.
   <p>
   * 
   */ public
 static boolean checkTerm(int sboTerm) {
    return libsbmlJNI.SBO_checkTerm__SWIG_1(sboTerm);
  }

  public SBO() {
    this(libsbmlJNI.new_SBO(), true);
  }

}
