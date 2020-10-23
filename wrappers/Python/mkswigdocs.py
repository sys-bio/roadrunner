#! /usr/bin/env python

"""
This program parses the rst documents in this directory, extracts the docstrings
and writes these docstrings into the roadrunner/rr_docstrings.i file which is
compiled into the roadrunner python module.
"""


# src format:
# .. method:: ExecutableModel.getModelName()
# .. py:method:: ???????????
# .. py:attribute:: RoadRunner.model

# dst format:
#%feature("docstring") rr::RoadRunner::getCC "
#getCC(variable, parameter)
#Get scaled a single control coefficient with respect to a global parameter.
#
#:param variable: The id of a dependent variable of the coefficient, for example a
#flux or species concentration.
#:type name: str
#:param parameter: The id of the independent parameter, for example a kinetic constant
#or boundary species
#:returns: the value of the control coefficeint returned to the caller.
#:rtype: double
#";


import sys
import re

def write_docstring(file, category, cls, name, args, lines):
    print(category, cls, name, args)

    def ltrim(s):
        line = s[3:] if s.startswith("   ") else s
        # yes this looks weird, but we do need to replace a backslash literal
        # with four of them. Evidently something to do with C need an extra one
        # to escape the escape, then need another pair to escape that when python
        # reads it.
        line = line.replace("\\", "\\\\\\\\") 
        line = line.replace('"', '\\"')
        return line

    # strips trailing emtpy lines
    index = len(lines) -1
    while(index > 0):
        if not len(lines[index].strip(' \t\n\r')):
            del lines[index]
            index = index - 1
        else:
            break

    lines = [ltrim(line.rstrip()) for line in lines if line.find(":module:") == -1]



    if name == "__init__":
        file.write("%feature(\"docstring\") rr::" + cls + " \"\n")
    else:
        file.write("%feature(\"docstring\") rr::" + cls + "::" + name + " \"\n")

    if category != "attribute":
        file.write(cls + "." + name + args + "\n")

    for line in lines:
        file.write(line + "\n")
    file.write("\";\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")


# method re
method = re.compile(r"\.\.\s*[py:]*(class|function|staticmethod|method)::\s*(\w+)\.(\w+)(\(.*\))")
attr = re.compile(r"\.\.\s*[py:]*(staticattribue|attribute)::\s*(\w+)\.(\w+)")

def read_docs(outFile, srcName):
    src = open(srcName, "r")
    lines = src.readlines()
    del src

    index = 0

    while index < len(lines):
        m = method.match(lines[index])
        a = attr.match(lines[index])
        if m:
            groups = m.groups()
            index = index + 1
            docLines = []
            while index < len(lines) and method.match(lines[index]) is None and attr.match(lines[index]) is None:
                docLines.append(lines[index])
                index = index + 1

            write_docstring(out, groups[0], groups[1], groups[2], groups[3], docLines)

        elif a:
            groups = a.groups()
            index = index + 1
            docLines = []
            while index < len(lines) and method.match(lines[index]) is None and attr.match(lines[index]) is None:
                docLines.append(lines[index])
                index = index + 1

            write_docstring(out, groups[0], groups[1], groups[2], None, docLines)

        else:
            index = index + 1




print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


result="roadrunner/rr_docstrings.i"

out = open(result, "w")

srcFiles = ["doc/mod_roadrunner/cls_ExecutableModel.rst",
            "doc/mod_roadrunner/cls_RoadRunner.rst",
            "doc/mod_roadrunner/cls_SelectionRecord.rst",
            "doc/mod_roadrunner/cls_SimulateOptions.rst", 
            "doc/mod_roadrunner/cls_LoadSBMLOptions.rst", 
            "doc/mod_roadrunner/cls_Logger.rst",
            "doc/mod_roadrunner/cls_Config.rst",
            "doc/mod_roadrunner/cls_PyConservedMoietyConverter.rst"]

for f in srcFiles:
    read_docs(out, f)









