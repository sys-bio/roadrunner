/**
 * Filename    : comp-package.i
 * Description : SBML comp swig file for bindings.
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
 * documentation provided hereunder is on an as is" basis, and the European
 * Media Laboratories gGmbH have no obligations to provide maintenance,
 * support, updates, enhancements or modifications.  In no event shall the
 * European Media Laboratory gGmbH be liable to any party for direct,
 * indirect, special, incidental or consequential damages, including lost
 * profits, arising out of the use of this software and its documentation,
 * even if the European Media Laboratories gGmbH have been advised of the
 * possibility of such damage.  See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 */

#ifdef USE_COMP


%ignore Submodel::getAllInstantiatedElements;

%extend Submodel
{
	ListWrapper<SBase>* getListOfAllInstantiatedElements()
	{
		List* list = $self->getAllInstantiatedElements();
		return new ListWrapper<SBase>(list);
	}
}

/* 
 * Currently we have to ignore methods that take std::list and vector.
 */
%ignore SBMLFileResolver::setAdditionalDirs(const std::vector<std::string>& dirs);


%include <sbml/packages/comp/extension/CompExtension.h>
%include <sbml/packages/comp/extension/CompSBasePlugin.h>
%include <sbml/packages/comp/extension/CompModelPlugin.h>
%include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>

%include sbml/packages/comp/util/SBMLUri.h
%include sbml/packages/comp/util/SBMLResolver.h
%include sbml/packages/comp/util/SBMLFileResolver.h
%include sbml/packages/comp/util/SBMLResolverRegistry.h
%include <sbml/packages/comp/util/CompFlatteningConverter.h>

%include sbml/packages/comp/sbml/CompBase.h
%include sbml/packages/comp/sbml/SBaseRef.h
%include sbml/packages/comp/sbml/Replacing.h
%include sbml/packages/comp/sbml/Deletion.h
%include sbml/packages/comp/sbml/ExternalModelDefinition.h
%include sbml/packages/comp/sbml/ListOfDeletions.h
%include sbml/packages/comp/sbml/ListOfExternalModelDefinitions.h
%include sbml/packages/comp/sbml/ListOfModelDefinitions.h
%include sbml/packages/comp/sbml/ListOfPorts.h
%include sbml/packages/comp/sbml/ListOfReplacedElements.h
%include sbml/packages/comp/sbml/ListOfSubmodels.h
%include sbml/packages/comp/sbml/ModelDefinition.h
%include sbml/packages/comp/sbml/Port.h
%include sbml/packages/comp/sbml/ReplacedBy.h
%include sbml/packages/comp/sbml/ReplacedElement.h
%include sbml/packages/comp/sbml/Submodel.h
%include sbml/packages/comp/sbml/Replacing.h

%include sbml/packages/comp/validator/CompSBMLError.h

%template(CompPkgNamespaces) SBMLExtensionNamespaces<CompExtension>;


#endif /* USE_COMP */
