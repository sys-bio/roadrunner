/**
 * @file    SBMLFileResolver.cpp
 * @brief   Implementation of SBMLFileResolver, the file system based resolver for SBML Documents.
 * @author  Frank Bergmann
 *
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2013-2014 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *     3. University of Heidelberg, Heidelberg, Germany
 * 
 * Copyright 2011-2012 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ------------------------------------------------------------------------ -->
 */

#include <sbml/packages/comp/util/SBMLFileResolver.h>
#include <sbml/packages/comp/util/SBMLUri.h>
#include <sbml/SBMLDocument.h>
#include <sbml/SBMLConstructorException.h>
#include <sbml/SBMLReader.h>

#ifdef __cplusplus

#include <locale>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;
LIBSBML_CPP_NAMESPACE_BEGIN

SBMLFileResolver::SBMLFileResolver ()
{
}


/*
 * Copy constructor.
 */
SBMLFileResolver::SBMLFileResolver(const SBMLFileResolver& orig)
{
  if (&orig == NULL)
  {
    throw SBMLConstructorException("Null argument to copy constructor");
  }
  else
  {
    setAdditionalDirs(orig.mAdditionalDirs);
  }
}


/*
 * Destroy this object.
 */
SBMLFileResolver::~SBMLFileResolver ()
{

}


/*
 * Assignment operator for SBMLFileResolver.
 */
SBMLFileResolver&
SBMLFileResolver::operator=(const SBMLFileResolver& rhs)
{
  if (&rhs == NULL)
  {
    throw SBMLConstructorException("Null argument to assignment operator");
  }
  else if(&rhs!=this)
  {
    setAdditionalDirs(rhs.mAdditionalDirs);
  }

  return *this;
}


SBMLFileResolver*
SBMLFileResolver::clone () const
{
  return new SBMLFileResolver(*this);
}


SBMLDocument*
SBMLFileResolver::resolve(const std::string &sUri, const std::string& sBaseUri/*=""*/) const
{
  // leave the actual work of resolving to the other function call that is 
  // needed for the caching mechanism in CompSBMLDocumentPlugin
  SBMLUri *uri = resolveUri(sUri, sBaseUri);
  if (uri == NULL)
    return NULL;
  
  const string fileName = uri->getPath();
  delete uri;

  if (fileExists(fileName))
    return readSBML(fileName.c_str());

  return NULL;

}

void prefixFileIfNeeded(std::string& fileName)
{
  if (fileName.length() == 0) return;
  if (fileName[0] == '/') return;
  fileName = "/" + fileName;
}

#if !defined(WIN32) || defined(CYGWIN)
#include <dirent.h>
bool directoryExists( const char* path )
{
  if ( path == NULL) return false;

  bool result = false;
  DIR *dir = opendir (path);


  if (dir != NULL)
  {
    result = true;    
    (void) closedir (dir);
  }

  return result;
}
#endif


SBMLUri* 
SBMLFileResolver::resolveUri(const std::string &sUri, const std::string& sBaseUri/*=""*/) const
{
  string fileName = sUri;


  // filter scheme
  SBMLUri uri(sUri);
  SBMLUri baseUri(sBaseUri);

  if (uri.getScheme() != "file" && baseUri.getScheme() != "file")
    return NULL;


  // also moved this as Lucian suggests that if there is a local file
  // it is not necessarily the one referenced
  // greedily pick up the first file we see before trying more
  // elaborate means
  //if (fileExists(fileName))
  //  return new SBMLUri(fileName);
    
    
  std::vector<std::string>::const_iterator it = mAdditionalDirs.begin();
  while(it != mAdditionalDirs.end())
  {
    fileName = SBMLUri(*it).relativeTo(uri.getPath()).getPath();  
    if (fileExists(fileName))
      return new SBMLUri(fileName);
    // missing root? 
    prefixFileIfNeeded(fileName);
    if (fileExists(fileName))
      return new SBMLUri(fileName);
    ++it;
  }
  

  // if we do this here we may end up adding a file name to a path 
  // that already contains a filename
  // which is obviously illogical
  // it should fail neatly and move on to try without the filename
  // but just to be totally careful lets move it to afterwards

  //fileName = baseUri.relativeTo(uri.getPath()).getPath();  
  //if (fileExists(fileName))
  //  return new SBMLUri(fileName);
  //// missing root?
  //prefixFileIfNeeded(fileName);
  //if (fileExists(fileName))
  //  return new SBMLUri(fileName);


  // adjust for baseUri, when invoked with *full* filename, rather than just the path
  fileName = baseUri.getPath();
  size_t pos = fileName.rfind('/');
  if (pos != fileName.npos)
  {
    fileName = SBMLUri(fileName.substr(0, pos)).relativeTo(uri.getPath()).getPath();
    if (fileExists(fileName))
      return new SBMLUri(fileName);
    // missing root?
    prefixFileIfNeeded(fileName);
    if (fileExists(fileName))
      return new SBMLUri(fileName);

  }

  fileName = baseUri.relativeTo(uri.getPath()).getPath();  
  if (fileExists(fileName))
    return new SBMLUri(fileName);
  // missing root?
  prefixFileIfNeeded(fileName);
  if (fileExists(fileName))
    return new SBMLUri(fileName);

  // well lets just check ...
  fileName = sUri;
  if (fileExists(fileName))
    return new SBMLUri(fileName);

  // we could scramble a bit more by continuing to search but for now that 
  // ought to be it.

  return NULL;
}

/** @cond doxygenLibsbmlInternal */
bool
SBMLFileResolver::fileExists(const std::string& fileName)
{
#if !defined(WIN32) || defined(CYGWIN)
  // cygwin has an issue if the filename is actually a directory
  // later attempts to create relative paths cause infinite loops
  // in the attempts to create files/open directories
  // thus if we are in cygwin - check if the name is a directory
  // and get out now if it is
  if (directoryExists(fileName.c_str()))
  {
    return false;
  }
#endif
  ifstream file(fileName.c_str());
  if (!file)
    return false;
  file.close();
  // on linux we know that ther fileName exists, however 
  // it could be a directory (windows does not allow 
  // a directory be opened)
  #if !defined(WIN32) || defined(CYGWIN)
  if (directoryExists(fileName.c_str()))
    return false;
  #endif
  return true;
}
/** @endcond */


void 
SBMLFileResolver::setAdditionalDirs(const std::vector<std::string>& dirs)
{
  mAdditionalDirs.assign(dirs.begin(), dirs.end());
}

void 
SBMLFileResolver::clearAdditionalDirs()
{
  mAdditionalDirs.clear();
}

void 
SBMLFileResolver::addAdditionalDir(const std::string& dir)
{
  mAdditionalDirs.push_back(dir);
}

/** @cond doxygenIgnored */


/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


