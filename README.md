#  Steps
1. Check the Output of the PDF from PyPDF2
1. If NO output then assume the pdf is an image.
    1. Pass it to the OCR engine and get the text output.
    1. Send the OCR output further.
1. If there is some Output then keep that output for further processing.