#pragma hdrstop
#include <float.h>
#include <stdarg.h>
#include <string>
#include <cstring>
#include <stdio.h>
#include "rrUtils.h"
#include "rrOSSpecifics.h"
#include "rrLogger.h"
#include "rrIniFile.h"

#undef CreateFile

namespace rr
{
//const bool IniFile::mAutoCreateSections = true;
//const bool IniFile::mAutoCreateKeys		= true;

IniFile::IniFile(const std::string& szFileName, bool autoLoad, bool autoSave)
: mCommentIndicators(";#")
, mEqualIndicator("=")
, mWhiteSpace(" \t\n\r")
, mIniFileName(szFileName)
, mIsDirty(false)
, mAutoSave(autoSave)
, mAutoCreateKeys(1)
, mAutoCreateSections(1)
, mFlags(0)
{
	//mFlags = (mAutoCreateSections | mAutoCreateKeys);
	if(mIniFileName.size() > 2 && autoLoad == true)
	{
		Load();	//Load all sections and keys upon creation...
		rrLog(lDebug3)<<"Loaded file: " << mIniFileName.Get();
	}
}

IniFile::~IniFile()
{
	if (mIsDirty && mAutoSave)
    {
		Save();
    }

	Clear();
}

// Resets the member variables to their defaults
void IniFile::Clear()
{
	mIsDirty = true;			//A save after clear, will clear the phuysica; file
	for(unsigned int i = 0; i < mSections.size(); i++)
	{
		IniSection* section = mSections[i];
		delete section;
	}
	mSections.clear();
}

bool IniFile::SetFilePath(const std::string& path)
{
	return mIniFileName.SetPath(path);
}

bool IniFile::ClearSection(const std::string& aSection)
{
	IniSection* section = GetSection(aSection);
	if(section)
	{
		section->Clear();
		return true;
	}
	return false;
}

// Set's the mFileName member variable. For use when creating the IniFile
// object by hand versus loading it from a file
void IniFile::SetFileName(const std::string& FileName)
{
	if (mIniFileName.size() != 0 && compareNoCase(FileName, mIniFileName) != 0)
	{
		mIsDirty = true;
	}

	mIniFileName = FileName;
}

// Attempts to load in the text file. If successful it will populate the
// Section list with the key/value pairs found in the file. Note that comments
// are saved so that they can be rewritten to the file later.
bool IniFile::Load(const std::string& newfName)
{
	if(newfName.size())
    {
    	mIniFileName = newfName;
    }

	if(mIniFileName.size() < 1)
	{
									rrLog(Logger::LOG_ERROR)<<"Ini file name is not set!";
		return false;
	}
	else
	{
									rrLog(lDebug3)<<"Loading IniFile: " <<mIniFileName.Get();
	}

    if(!std::filesystem::exists(mIniFileName.Get()))
    {
									rrLog(lDebug2)<<"The ini file: \"" <<mIniFileName.Get()<<"\" does not exist. It will be created";
        if(!createFile(mIniFileName))
        {
    								rrLog(Logger::LOG_ERROR)<<"Failed to create ini file.. returning..";
            return false;
        }
	}

	std::string fName(mIniFileName);
    std::fstream file;
	file.open(fName.c_str(), std::ios::in|std::ios::out);
	if(file.fail())
    {
    								rrLog(Logger::LOG_ERROR)<<"Opening file caused failbit to be set";
    }

	if (!file.is_open() )
	{
									rrLog(Logger::LOG_ERROR)<<"The ini file: '" <<mIniFileName<<"' could not be opened";
		return false;
	}
	else
	{
		bool bDone = false;
		bool bAutoKey = (mFlags & mAutoCreateKeys) 		== mAutoCreateKeys;
		bool bAutoSec = (mFlags & mAutoCreateSections) 	== mAutoCreateSections;
		IniSection* pSection = GetSection("");

		// These need to be set, we'll restore the original values later.
		mFlags |= mAutoCreateKeys;
		mFlags |= mAutoCreateSections;

		char* buffer = new char[MAX_LINE_BUFFER_SIZE];
        int lines = 0;
		while(!bDone)
		{
			std::string Comment;
			memset(buffer, 0, MAX_LINE_BUFFER_SIZE);
			file.getline(buffer, MAX_LINE_BUFFER_SIZE);
			lines++;

			std::string Line = buffer;
			Trim(Line);

			bDone = ( file.eof() || file.bad() || file.fail() );

			if ( Line.find_first_of(mCommentIndicators) == 0 )
			{
				Comment += "\n";
				Comment += Line;
			}
			else if ( Line.find_first_of('[') == 0 ) // Found a section
			{
				Line.erase( 0, 1 );
				Line.erase( Line.find_last_of(']'), 1 );
				pSection = GetSection(Line, true);
            	rrLog(lDebug3)<<"Located section: " + pSection->mName;
				Comment = std::string("");
            }
            else if ( Line.find_first_of('<') != std::string::npos &&
                      Line.find_first_of('>') != std::string::npos)     //XML
            {
                if(pSection)
                {
                    WriteNonKey(Line, pSection->mName);
                }
                rrLog(lDebug5)<<"Read NonKey on line: " << lines;
            }
            else if ( Line.size() > 0 ) // we have a key, add this key/value std::pair
			{
                std::string Comment = "";
                std::string Key = "";
                std::string Value = "";

            	std::vector<std::string> record = splitString(Line, mEqualIndicator + mCommentIndicators);
                if(record.size() > 1)
				{
	                std::string Comment = "";
                	std::string Key = Trim(record[0]);
                    std::string Value = Trim(record[1]);
                    if(record.size() > 2)
                    {
                    	Comment = record[2];
                    }

					if(pSection)
					{
						WriteValue(Key, Value, Comment, pSection->mName);
						rrLog(lDebug5)<<Key << " = "  <<Value;
					}
					else
					{
                		rrLog(lDebug5)<<Key << " = " << Value;
						rrLog(lWarning)<<"No section for key"<<Key<<". Key was ignored..";
					}
					Comment = std::string("");
				}
				else	//Not a valid Key. Store as a NonKey
				{
					if(pSection)
					{
						WriteNonKey(Line, pSection->mName);
					}
					rrLog(lDebug5)<<"Read NonKey on line: " << lines;
				}
			}
		}

        delete [] buffer;
		// Restore the original flag values.
		if(!bAutoKey)
        {
			mFlags &= ~mAutoCreateKeys;
        }

		if(!bAutoSec)
        {
			mFlags &= ~mAutoCreateSections;
        }
	}
	file.close();
	return true;
}

IniSection* IniFile::LoadSection(const std::string& theSection)
{
    IniSection* pSection = NULL;
	if(mIniFileName.size() < 1)
	{
									rrLog(Logger::LOG_ERROR)<<"Ini file name is not set!";
		return NULL;
	}
	else
	{
									rrLog(lDebug3)<<"Loading IniFile: " <<mIniFileName.Get();
	}

    std::fstream file;
    if(!std::filesystem::exists(mIniFileName.Get()))
    {
									rrLog(lDebug2)<<"The ini file: \"" <<mIniFileName.Get()<<"\" does not exist. It will be created";
	}

	std::string fName(mIniFileName);
	file.open(fName.c_str(), std::ios::in|std::ios::out);
	if(file.fail())
    {
    								rrLog(Logger::LOG_ERROR)<<"Opening file caused failbit to be set";
    }

	if (!file.is_open() )
	{
									rrLog(Logger::LOG_ERROR)<<"The ini file: '" <<mIniFileName<<"' could not be opened";
		return NULL;
	}
	else
	{
		bool bDone = false;
		bool bAutoKey = (mFlags & mAutoCreateKeys) 		== mAutoCreateKeys;
		bool bAutoSec = (mFlags & mAutoCreateSections) 	== mAutoCreateSections;

		char* buffer = new char[MAX_LINE_BUFFER_SIZE];
		pSection = GetSection("");

		// These need to be set, we'll restore the original values later.
		mFlags |= mAutoCreateKeys;
		mFlags |= mAutoCreateSections;

		while(!bDone)
		{
			std::string Comment;
			memset(buffer, 0, MAX_LINE_BUFFER_SIZE);
			file.getline(buffer, MAX_LINE_BUFFER_SIZE);

			std::string Line = buffer;
			Trim(Line);

			bDone = ( file.eof() || file.bad() || file.fail() );

			if ( Line.find_first_of(mCommentIndicators) == 0 )
			{
				Comment += "\n";
				Comment += Line;
			}
			else
			if ( Line.find_first_of('[') == 0 ) // Found a section
			{
				Line.erase( 0, 1 );
				Line.erase( Line.find_last_of(']'), 1 );

                if(theSection == Line)
                {
    				CreateSection(Line, Comment);
	    			pSection = GetSection(Line);
										rrLog(lDebug3)<<"Located ini section: " + pSection->mName;
				    Comment = std::string("");
                    bDone = true;
                }
			}
			else if ( Line.size() > 0 ) // we have a key, add this key/value std::pair
			{
				std::string Key = GetNextWord(Line);
				std::string Value = Line;

				if ( Key.size() > 0 && Value.size() > 0 )
				{
					if(pSection)
					{
						WriteValue(Key, Value, Comment, pSection->mName);
										rrLog(lDebug5)<<"Read Key " + Key + " Value = " + Value;
					}
					else
					{
										rrLog(lDebug5)<<"Read Key " + Key + " Value = " + Value;
										rrLog(lWarning)<<"No section for key" + Key + ". Key was ignored..";
					}
					Comment = std::string("");
				}
				else	//Not a valid Key. Store as a NonKey
				{
					if(pSection)
					{
						WriteNonKey(Key, pSection->mName);
						rrLog(lDebug5)<<"Read a NonKey: " + Key;
					}
				}
			}
		}

		// Restore the original flag values.
		if (!bAutoKey)
		{
			mFlags &= ~mAutoCreateKeys;
		}

		if (!bAutoSec)
		{
			mFlags &= ~mAutoCreateSections;
		}
	}

	file.close();
	return pSection;
}

// Attempts to save the Section list and keys to the file. Note that if Load
// was never called (the IniFile object was created manually), then you
// must set the mFileName variable before calling save.
bool IniFile::Save(ios_base::openmode openMode)
{
	if ( mIniFileName.size() == 0 )
	{
		rrLog(Logger::LOG_ERROR)<<"No filename has been set. Can't save..";
		return false;
	}

    std::string aFName = mIniFileName.Get();
	std::fstream aFstream(aFName.c_str(), openMode);

	if(aFstream.is_open())
	{
		SectionItor s_pos;
		KeyItor k_pos;
		IniSection* Section;
		IniKey* Key;

		for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
		{
			Section = (*s_pos);
			bool bWroteComment = false;

			if ( Section->mComment.size() > 0 )
			{
				bWroteComment = true;
				WriteLine(aFstream, "\n%s", CommentStr(Section->mComment).c_str());
			}

			if ( Section->mName.size() > 0 )
			{
				WriteLine(aFstream, "%s[%s]",
						bWroteComment ? "" : "\n",
						Section->mName.c_str());
			}

			for (k_pos = Section->mKeys.begin(); k_pos != Section->mKeys.end(); k_pos++)
			{
				Key = (*k_pos);

				if ( Key->mKey.size() > 0 && Key->mValue.size() > 0 )
				{
						WriteLine(aFstream, "%s%s%s%s%c%s",
						Key->mComment.size() > 0 ? "\n" : "",
						CommentStr(Key->mComment).c_str(),
						Key->mComment.size() > 0 ? "\n" : "",
						Key->mKey.c_str(),
						mEqualIndicator[0],
						Key->mValue.c_str());
				}
			}
		}
	}
	else
	{
		rrLog(Logger::LOG_ERROR)<<"[IniFile::Save] Unable to save file.";
		return false;
	}

	mIsDirty = false;
	aFstream.flush();
	aFstream.close();
								rrLog(lDebug3)<<"IniFile was saved";
	return true;
}

// Set the comment of a given key. Returns true if the key is not found.
bool IniFile::SetKeyComment(const std::string& mKey, const std::string& mComment, const std::string& szSection)
{
	KeyItor k_pos;
	IniSection* pSection;

	if ( (pSection = GetSection(szSection)) == NULL )
		return false;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( compareNoCase( (*k_pos)->mKey, mKey ) == 0 )
		{
			(*k_pos)->mComment = mComment;
			mIsDirty = true;
			return true;
		}
	}

	return false;
}

// Set the comment for a given section. Returns false if the section
// was not found.
bool IniFile::SetSectionComment(const std::string& Section, const std::string& Comment)
{
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
    	std::string name = (*s_pos)->mName;
		if ( compareNoCase( name , Section ) == 0 )
		{
		    (*s_pos)->mComment = Comment;
			mIsDirty = true;
			return true;
		}
	}

	return false;
}

bool IniFile::SectionExists(const std::string& section)
{
	SectionItor s_pos;
	IniSection* Section;

    for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
			Section = (*s_pos);
            if(Section->mName == section)
            	return true;
    }
	return false;
}

// Given a key, a value and a section, this function will attempt to locate the
// Key within the given section, and if it finds it, change the keys value to
// the new value. If it does not locate the key, it will create a new key with
// the proper value and place it in the section requested.
bool IniFile::WriteValue(const std::string& mKey, const std::string& mValue, const std::string& mComment, const std::string& szSection)
{
	IniKey* 		pKey 		= GetKey(mKey, szSection);
	IniSection* 	pSection 	= GetSection(szSection);

	if (pSection == NULL)
	{
		if ( !(mFlags & mAutoCreateSections) || !CreateSection(szSection))
			return false;

		pSection = GetSection(szSection);
	}

	// Sanity check...
	if ( pSection == NULL )
		return false;

	// if the key does not exist in that section, and the value passed
	// is not std::string("") then add the new key.
	if ( pKey == NULL && mValue.size() > 0 && (mFlags & mAutoCreateKeys))
	{
		pKey = new IniKey;

		pKey->mKey = mKey;
		pKey->mValue = mValue;
		pKey->mComment = mComment;
		mIsDirty = true;
		pSection->mKeys.push_back(pKey);
		return true;
	}

	if ( pKey != NULL )
	{
		pKey->mValue = mValue;
		pKey->mComment = mComment;
		mIsDirty = true;
		return true;
	}

	return false;
}

bool IniFile::WriteNonKey(const std::string& nonKey, const std::string& section)
{
	IniSection* 	pSection 	= GetSection(section);
	if (pSection == NULL)
	{
		if ( !(mFlags & mAutoCreateSections) || !CreateSection(section))
			return false;

		pSection = GetSection(section);
	}

	// Sanity check...
	if (pSection == NULL )
		return false;

    pSection->mNonKeys.push_back(nonKey);
	return true;
}

// Passes the given float to WriteValue as a std::string
bool IniFile::WriteFloat(const std::string& mKey, double value, const std::string& mComment, const std::string& szSection)
{
	char szStr[64];
	snprintf(szStr, 64, "%g", value);
	return WriteValue(mKey, szStr, mComment, szSection);
}

// Passes the given int to WriteValue as a std::string
bool IniFile::WriteInteger(const std::string& mKey, int nValue, const std::string& mComment, const std::string& szSection)
{
	char szStr[64];
	snprintf(szStr, 64, "%d", nValue);
	return WriteValue(mKey, szStr, mComment, szSection);
}

// Passes the given bool to WriteValue as a std::string
bool IniFile::WriteBool(const std::string& mKey, bool bValue, const std::string& mComment, const std::string& szSection)
{
	std::string mValue = bValue ?  "true" : "false";

	return WriteValue(mKey, mValue, mComment, szSection);
}

// Returns the key value as a std::string object. A return value of
// std::string("") indicates that the key could not be found.
std::string IniFile::ReadValue(const std::string& mKey, const std::string& szSection)
{
	IniKey* pKey = GetKey(mKey, szSection);
  	mWasFound = pKey ? true : false;

    std::string value;
    if(pKey)
    {
        value = pKey->mValue;
    }
    else
    {
    	value = "";
    }

	return value;
}

// Returns the key value as a std::string object. A return value of
// std::string("") indicates that the key could not be found.
std::string IniFile::ReadString(const std::string& mKey, const std::string& szSection, const std::string& def_val)
{
	std::string 	str =  ReadValue(mKey, szSection);
	if(mWasFound)
	{
		return str;
	}
	return def_val;
}

// Returns the key value as a double type. Returns FLT_MIN if the key is
// not found.
double IniFile::ReadDouble(const std::string& mKey, const std::string& szSection, double def_value)
{
	std::string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		return atof( mValue.c_str() );
	}

	return def_value;
}

// Returns the key value as an integer type. Returns INT_MIN if the key is
// not found.
int	IniFile::ReadInteger(const std::string& mKey, const std::string& szSection, int def_val)
{
	std::string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		return	atoi(mValue.c_str());
	}
	return def_val;
}

// Returns the key value as a bool type. Returns false if the key is
// not found.
bool IniFile::ReadBool(const std::string& mKey, const std::string& szSection, bool def_value)
{
	bool bValue = def_value;
	std::string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		if ( mValue.find("1") 						== 0 	
				|| compareNoCase(mValue, "true") 	== 0 
				|| compareNoCase(mValue, "yes") 	== 0 )
		{
			bValue = true;
		}
        else
        {
        	bValue = false;
        }
	}	

	return bValue;
}

// Delete a specific section. Returns false if the section cannot be
// found or true when sucessfully deleted.
bool IniFile::DeleteSection(const std::string& Section)
{
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
		if ( compareNoCase( (*s_pos)->mName, Section ) == 0 )
		{
        	//Found a section..
            IniSection *aSection =  (*s_pos);
			mSections.erase(s_pos);
            delete (aSection);
			return true;
		}
	}

	return false;
}

bool IniFile::DeleteSectionsWithKeyValue(const std::string& keyName, const std::string& value)
{
	//First find sections containing the key
	SectionItor s_pos = mSections.begin();
    s_pos = mSections.begin();
	while (s_pos != mSections.end())
	{
		IniKey* key = GetKey(keyName, (*s_pos)->mName);
        {
        	if(key && key->mValue == value)
            {
           		//remove section
              	DeleteSection((*s_pos)->mName);
			    s_pos = mSections.begin();
            }
            else
            {
            	++s_pos;
            }
        }
    }
	return true;
}

// Delete a specific key in a specific section. Returns false if the key
// cannot be found or true when sucessfully deleted.
bool IniFile::DeleteKey(const std::string& Key, const std::string& FromSection)
{
	KeyItor k_pos;
	IniSection* pSection;

	if ( (pSection = GetSection(FromSection)) == NULL )
		return false;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( compareNoCase( (*k_pos)->mKey, Key ) == 0 )
		{
        	IniKey* aKey = (*k_pos);
			pSection->mKeys.erase(k_pos);
            delete aKey;
			return true;
		}
	}

	return false;
}

// Given a key, a value and a section, this function will attempt to locate the
// Key within the given section, and if it finds it, change the keys value to
// the new value. If it does not locate the key, it will create a new key with
// the proper value and place it in the section requested.
bool IniFile::CreateKey(const std::string& mKey, const std::string& mValue, const std::string& mComment, const std::string& szSection)
{
	bool bAutoKey = (mFlags & mAutoCreateKeys) == mAutoCreateKeys;
	bool bReturn  = false;

	mFlags |= mAutoCreateKeys;

	bReturn = WriteValue(mKey, mValue, mComment, szSection);

	if (!bAutoKey)
	{
		mFlags &= ~mAutoCreateKeys;
	}

	return bReturn;
}

// Given a section name, this function first checks to see if the given section
// allready exists in the list or not, if not, it creates the new section and
// assigns it the comment given in mComment.  The function returns true if
// sucessfully created, or false otherwise.
bool IniFile::CreateSection(const std::string& Section, const std::string& mComment)
{
	IniSection* pSection = GetSection(Section);

	if ( pSection )
	{
		rrLog(lDebug5)<<"[IniFile::CreateSection] Section "<<Section.c_str()<<" already exists. Aborting.";
		return false;
	}

	pSection = new IniSection;
	pSection->mName = Section;
	pSection->mComment = mComment;
	mSections.push_back(pSection);
	mIsDirty = true;
	return true;
}

// Given a section name, this function first checks to see if the given section
// allready exists in the list or not, if not, it creates the new section and
// assigns it the comment given in mComment.  The function returns true if
// sucessfully created, or false otherwise. This version accpets a KeyList
// and Writes up the newly created Section with the keys in the list.
bool IniFile::CreateSection(const std::string& Section, const std::string& Comment, KeyList Keys)
{
	if ( !CreateSection(Section, Comment) )
		return false;

	IniSection* pSection = GetSection(Section);

	if ( !pSection )
		return false;

	KeyItor k_pos;

	pSection->mName = Section;
	for (k_pos = Keys.begin(); k_pos != Keys.end(); k_pos++)
	{
		IniKey* pKey = new IniKey;
		pKey->mComment = (*k_pos)->mComment;
		pKey->mKey = (*k_pos)->mKey;
		pKey->mValue = (*k_pos)->mValue;
		pSection->mKeys.push_back(pKey);
	}

	mSections.push_back(pSection);
	mIsDirty = true;

	return true;
}

// Simply returns the number of sections in the list.
size_t IniFile::SectionCount()
{
	return mSections.size();
}

// Returns the total number of keys contained within all the sections.
size_t IniFile::KeyCount()
{
	size_t nCounter = 0;
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
		nCounter += (*s_pos)->mKeys.size();

	return nCounter;
}

size_t IniFile::KeyCount(const std::string& section)
{
	//Get the section
    IniSection* iniSection = GetSection(section);
    return iniSection->KeyCount();
}

// Given a key and section name, looks up the key and if found, returns a
// pointer to that key, otherwise returns NULL.
IniKey*	IniFile::GetKey(const std::string& Key, const std::string& Section)
{
	IniSection* pSection;
	KeyItor k_pos;

	// Since our default section has a name value of std::string("") this should
	// always return a valid section, wether or not it has any keys in it is
	// another matter.
	if ( (pSection = GetSection(Section)) == NULL )
		return NULL;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( compareNoCase( (*k_pos)->mKey, Key ) == 0 )
			return (*k_pos);
	}

	return NULL;
}

// Given a section name, locates that section in the list and returns a pointer
// to it. If the section was not found, returns NULL
IniSection* IniFile::GetSection(const std::string& Section, bool create)
{
	SectionItor s_pos;
	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
    	if(s_pos == mSections.end())
        {
        	return NULL;
        }
        std::string secName = (*s_pos)->mName;
		if ( compareNoCase( secName, Section ) == 0 )
        {
        	IniSection* sec = (*s_pos);
			return sec;
        }
	}

    if(create)
    {
        CreateSection(Section);
        return GetSection(Section, false);
    }

	return NULL;
}

IniSection* IniFile::GetSection(const unsigned int sectionNr)
{
	if(sectionNr < mSections.size() && mSections.size() > 0)
    	return mSections[sectionNr];

	return NULL;
}

std::string IniFile::CommentStr(std::string& mComment)
{
	std::string szNewStr = std::string("");

	Trim(mComment);

		if ( mComment.size() == 0 )
		  return mComment;


	if ( mComment.find_first_of(mCommentIndicators) != 0 )
	{
		szNewStr = mCommentIndicators[0];
		szNewStr += " ";
	}

	szNewStr += mComment;

	return szNewStr;
}

// Given a key+delimiter+ value std::string, pulls the key name from the std::string,
// deletes the delimiter and alters the original std::string to contain the
// remainder.  Returns the key
std::string IniFile::GetNextWord(std::string& CommandLine)
{
	size_t nPos = CommandLine.find_first_of(mEqualIndicator);
	std::string sWord = std::string("");

	if ( nPos != std::string::npos )
	{
		sWord = CommandLine.substr(0, nPos);
		CommandLine.erase(0, nPos+1);
	}
	else
	{
		sWord = CommandLine;
		CommandLine = std::string("");
	}

	Trim(sWord);
	return sWord;
}

std::string IniFile::Trim(std::string& str)
{
	std::string szTrimChars 	= mWhiteSpace;
	szTrimChars 		+= mEqualIndicator;

	// Trim Both leading and trailing spaces
	size_t startpos = str.find_first_not_of(szTrimChars); 	// Find the first character position after excluding leading blank spaces
	size_t endpos 	= str.find_last_not_of(szTrimChars); 	// Find the first character position from reverse af

	// if all spaces or empty return an empty std::string
	if(( std::string::npos == startpos ) || ( std::string::npos == endpos))
	{
		str = "";
	}
	else
    {
		str = str.substr(startpos, endpos-startpos + 1);
    }

	return str;
}

// Writes the formatted output to the file stream, returning the number of
// bytes written.
int IniFile::WriteLine(std::fstream& stream, const char* fmt, ...)
{
	char *buf = new char[MAX_LINE_BUFFER_SIZE];
	int nLength;
	std::string szMsg;

	memset(buf, 0, MAX_LINE_BUFFER_SIZE);
	va_list args;

	va_start (args, fmt);
	  nLength = vsnprintf(buf, MAX_LINE_BUFFER_SIZE, fmt, args);
	va_end (args);


	if ( buf[nLength] != '\n' && buf[nLength] != '\r' )
		buf[nLength++] = '\n';

	stream.write(buf, nLength);

	delete [] buf;
	return nLength;
}


}
