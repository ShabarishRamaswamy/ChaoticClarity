import fitz
import PyPDF2

pdffile = "./pdfs/1.pdf"
doc = fitz.open(pdffile)

for i in range(1000):
    try:
        print(i)
        page = doc.loadPage(i)
        pix = page.getPixmap()
        output = "../outputs/outfile" + str(i) + ".png"
        pix.writePNG(output)
    except ValueError:
        break
print("Done !")

FILE_PATH = './pdfs/TU.pdf'

with open(FILE_PATH, mode='rb') as f:
    reader = PyPDF2.PdfFileReader(f)
    page = reader.getPage(10)
    print(page.extractText())