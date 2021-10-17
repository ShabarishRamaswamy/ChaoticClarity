#  Steps
1. Check the Output of the PDF from PyPDF2
1. If NO output then assume the pdf is an image.
    1. Pass it to the OCR engine and get the text output.
    1. Send the OCR output further.
1. If there is some Output then keep that output for further processing.
1. Send output text for summarization - the resultant file is saved as summary.txt
1. Next the original text is parsed by two NER models - blackstone and lexnlp for identifying important terms in every line.
1. Output of the above is saved in terms.txt and insg.txt respectively.
