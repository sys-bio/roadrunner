#! /usr/bin/bash

echo "Creating documentation for the plugin"
mainDoc=mainDoc.tex
pluginName=lmfit
chapter=plugin_$pluginName
pdflatex -jobname=$pluginName "\includeonly{$chapter}\input{$mainDoc}"

##Create the C file to link with the plugin
hdrFile="$pluginName"_doc.h
cppFile="$pluginName"_doc.cpp  

#Create header
echo "#ifndef $pluginName"_docH > $hdrFile
echo "#define $pluginName"_docH >> $hdrFile
echo "#include <stddef.h>" >> $hdrFile
echo "extern unsigned char pdf_doc[];" >> $hdrFile
echo "extern size_t sizeofPDF;" >> $hdrFile
echo "#endif" >> $hdrFile

#create source
echo "#include \"$hdrFile\"" > $cppFile
echo "" >> $cppFile
echo "unsigned char pdf_doc[]={" >> $cppFile

cat $pluginName.pdf | (xxd -i; echo "};") >> $cppFile
echo "size_t sizeofPDF = sizeof(pdf_doc);" >> $cppFile

cp $hdrFile $cppFile ..
echo "Done.."


