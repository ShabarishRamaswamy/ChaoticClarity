import fitz
import PyPDF2
import pytesseract
from PIL import Image


FILE_PATH = './pdfs/TU.pdf'
doc = fitz.open(FILE_PATH)


with open(FILE_PATH, mode='rb') as f:
    reader = PyPDF2.PdfFileReader(f)
    page = reader.getPage(1)
    print(page.extractText())
    if(page):
        print(page)
        print("UTF-8")
        pass
    else:
        read_from_ocr()

def read_from_ocr():
    for i in range(1000):
        try:
            page = doc.loadPage(i)
            pix = page.getPixmap()
            output = "outputs/outfile" + str(i) + ".png"
            pix.writePNG(output)
        except ValueError:
            break
    print("Done !")

    a = pytesseract.image_to_string(Image.open("./outputs/scr1.png"), lang="eng")
    print(a)