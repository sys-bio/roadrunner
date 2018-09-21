Conserved Moiety Conversion
___________________________


.. class:: PyConservedMoietyConverter()
   :module: RoadRunner

   Create a new conserved moiety converter. 

.. method:: PyConservedMoietyConverter.setDocument(sbmlOrURI)
   :module: RoadRunner

   Sets the source document. This may be either the contents of an sbml docment
   as a string, or the path the the file.

   :param str sbmlOrURI: contents or path of source document
   :returns: an integer indicating success or failure, 0 means success.
   :rtype: int


.. method:: PyConservedMoietyConverter.convert()
   :module: RoadRunner

   Perform the document conversion.

   :returns: an integer indicating success or failure, 0 means success.
   :rtype: int



.. method:: PyConservedMoietyConverter.getDocument()
   :module: RoadRunner

   Get the converted document contents.

   :returns: The contents of the converted document, or empty string
             if there is no source document.



