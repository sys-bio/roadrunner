#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=nelder_mead

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh
