#ifndef rrFileNameH
#define rrFileNameH
#include <string>
#include <ostream>
#include "rrExporter.h"
using std::string;
using std::ostream;

namespace rr
{

/**
 * @internal
 * Used in FileName.
 */
class RR_DECLSPEC FileName
{
    private:
        mutable std::string              mPathAndName;
        std::string                      mPath;
        std::string                      mName;
        void                        MakeFileString();

    public:
                                    FileName(const std::string& name = "", const std::string& path = "");
                                    FileName(const FileName& fN);
                                   ~FileName();

        FileName&                   operator = (const FileName& fN);
        FileName&                   operator = (const std::string& fN);
        FileName&                   operator = (const char* fN);
        bool                        operator !=(const char* fN);
                                    operator std::string();// {return mPathAndName;}
        bool                        SetPath(const std::string& path);
        void                        SetFileName(const std::string& name);
        void                        SetNameAndPath(const std::string& path, const std::string& name);
        void                        SetFileNameAndPath(const std::string& name);

        std::string                      GetFileName() const;
        std::string                      GetPath() const;
        std::string                      GetPathAndFileName() const;
        std::string                      Get();
        std::string                      GetFileNameNoExtension();
        size_t                      size();
};

RR_DECLSPEC std::ostream& operator         <<(std::ostream &os, FileName &obj);

}
#endif
