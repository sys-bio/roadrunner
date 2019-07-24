#! /usr/bin/bash

scripts="./../../../../scripts"
plugin=test_model

$scripts/makePDFDocs.sh $plugin
$scripts/makeHTMLDocs.sh $plugin

$scripts/cleanDocsFolder.sh
