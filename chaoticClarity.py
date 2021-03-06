# from app import send_ok
import fitz
import PyPDF2
import pytesseract
import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import networkx as nx
# nltk.download('stopwords')    
# nltk.download('punkt')
import spacy
from spacy import displacy
from blackstone.displacy_palette import ner_displacy_options
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import urllib
import pandas as pd
from requests_html import HTML
from requests_html import HTMLSession
# from flask import Flask
# app = Flask(__name__)
import os



PDF_REPO = "../ChaoticClarity-Backend/pdfs/"
OUTPUT_REPO = "../ChaoticClarity-Backend/outputs/"
MAIN_PDF_NAME = os.getenv("MAIN_PDF_NAME") or 0
TEXT_BASED_PDF = 1
testing = False


def sendPDFName(PDFName):
  print("PDFname: " + PDFName)
  global MAIN_PDF_NAME 
  MAIN_PDF_NAME = PDFName
  print("Inside SendPDF function: " + str(MAIN_PDF_NAME))
  return

if testing:
  if MAIN_PDF_NAME:
    pass
  else:
    sendPDFName("test-1")



print("Main PDF NAME: " + MAIN_PDF_NAME)

##
FILE_PATH = PDF_REPO + MAIN_PDF_NAME + ".pdf"

## LOOP for each page
with open(FILE_PATH, mode='rb') as f:
  reader = PyPDF2.PdfFileReader(f)

  for pages_number in range(5):
    try:
      page = reader.getPage(pages_number)
      page_contents = page.extractText()
      if page_contents:
        for line in page_contents:
          if line == "\n":
            TEXT_BASED_PDF = 0 # I.e Image based PDF.
            # print("Image Based PDF")
      else:
        TEXT_BASED_PDF = 0
        # print("Image Based PDF")

    except:
      continue


if TEXT_BASED_PDF == 0:
  doc = fitz.open(FILE_PATH)
  for i in range(doc.pageCount): # doc.pageCount
      try:
          page = doc.load_page(i)
          pix = page.get_pixmap()
          output = OUTPUT_REPO + MAIN_PDF_NAME + str(i) + ".png"
          # print(output)
          pix.save(output)
      except ValueError:
          break
  print("Done !")



## PyPDF2
## If No O/P, assume page is in IMG form.

## Tesseract.

# Continue.


with open( OUTPUT_REPO + "summary.txt", "a") as text_file:
    text_file.write("Summary:")

if TEXT_BASED_PDF == 0:
  doc = fitz.open(FILE_PATH)
  page_count = -1
  for i in range(doc.pageCount): # doc.pageCount
    page_count += 1
    print(str(i) + " Processing starts")

    ##
    from PIL import Image
    image_page_name = OUTPUT_REPO + MAIN_PDF_NAME + str(i) + ".png"
    ocr_output = pytesseract.image_to_string(Image.open(image_page_name), lang="eng")
    print(ocr_output)

    ##
    def cleaner(text):        
      sentences =[]        
      sentences = sent_tokenize(text)    
      for s in range(len(sentences)):
        ss = sentences[s]
        ss = re.sub(r'/.+?/', '', ss)
        ss = re.sub("[\(\[].*?[\)\]]", "", ss)
        sentences[s] = ss                      
      #print("\n".join(map(str,sentences)))
      #print("\n\n\n")
      return sentences


    def sim_matrix(sentences,stop_words):
      similarity_matrix = np.zeros((len(sentences),len(sentences)))
      for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
          if idx1!=idx2:
            similarity_matrix[idx1][idx2] = cosfunction(sentences[idx1],sentences[idx2],stop_words)
      return similarity_matrix


    def cosfunction(s1,s2,stopwords=None):    
      if stopwords is None:        
        stopwords = []        
      s1 = [w.lower() for w in s1]    
      s2 = [w.lower() for w in s2]
            
      all_words = list(set(s1 + s2))   
        
      vector1 = [0] * len(all_words)    
      vector2 = [0] * len(all_words)        
      for w in s1:        
        if not w in stopwords:
          vector1[all_words.index(w)]+=1                                                             
      for w in s2:        
        if not w in stopwords:            
          vector2[all_words.index(w)]+=1 
                  
      return 1-cosine_distance(vector1,vector2)

    def generate_summary(text,top_n):
      stop_words = stopwords.words('english')    
      summary = []
      sentences = cleaner(text)
      if len(sentences) > 6:
        cos_matrix = sim_matrix(sentences,stop_words)
        cos_graph = nx.from_numpy_array(cos_matrix)
        scores = nx.pagerank(cos_graph)
        ranks = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
        
        for i in range(top_n):
          summary.append(ranks[i][1])
        return " ".join(summary),len(sentences)
      return " "


    ##
    ocr_output = ocr_output.replace('\n', " ")
    print("OCR: ---- " + ocr_output)

    ## 
    g = generate_summary(ocr_output,4)[0].strip('\n')
    with open( OUTPUT_REPO + "summary.txt", "a" ) as text_file:
        text_file.write("%s" % g)
    print(g)


    ##
    bb = ocr_output.split('.')
    print(bb)

    ##

    # Load the model
    nlp = spacy.load("en_blackstone_proto")
    with open( OUTPUT_REPO + "terms.txt", "a") as text_file:
        text_file.write("Important terms: \n")
    for i in bb:
        doc = nlp(i)
    # Call displacy and pass `ner_displacy_options` into the option parameter`
        if doc.ents != ():
          print(doc.ents[0])
          with open(OUTPUT_REPO + "terms.txt", "a") as text_file:
            text_file.write("%s \n" %doc.ents[0])
        else:
          pass



    ##
    from lexnlp.extract.en.acts import get_acts as aa
    from lexnlp.extract.en.amounts import get_amounts as ab
    from lexnlp.extract.en.citations import get_citations as ac
    from lexnlp.extract.en.conditions import get_conditions as ae
    from lexnlp.extract.en.constraints import get_constraints as af
    from lexnlp.extract.en.copyright import get_copyright as ag
    from lexnlp.extract.en.courts import get_courts as ah
    from lexnlp.extract.en.dates import get_dates as ai
    from lexnlp.extract.en.definitions import get_definitions as aj
    from lexnlp.extract.en.distances import get_distances as ak
    from lexnlp.extract.en.durations import get_durations as al
    from lexnlp.extract.en.geoentities import get_geoentities as am
    from lexnlp.extract.en.money import get_money as an
    from lexnlp.extract.en.percents import get_percents as ao
    from lexnlp.extract.en.ratios import get_ratios as aq
    from lexnlp.extract.en.regulations import get_regulations as ar
    from lexnlp.extract.en.trademarks import get_trademarks as au

    with open( OUTPUT_REPO + "insg.txt", "w") as text_file:
        text_file.write("INSIGHTS: \n")

    for i in bb:
        if list(aa(i)) != []:
          print(list(aa(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(aa(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ab(i)) != []:
          print(list(ab(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ab(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ac(i)) != []:
          print(list(ac(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ac(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ae(i)) != []:
          print(list(ae(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ae(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(af(i)) != []:
          print(list(af(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(af(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ag(i)) != []:
          print(list(ag(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ag(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
      
        elif list(ai(i)) != []:
          print(list(ai(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ai(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(aj(i)) != []:
          print(list(aj(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(aj(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ak(i)) != []:
          print(list(ak(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ak(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(al(i)) != []:
          print(list(al(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(al(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        
        elif list(an(i)) != []:
          print(list(an(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(an(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ao(i)) != []:
          print(list(ao(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ao(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(aq(i)) != []:
          print(list(aq(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(aq(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(ar(i)) != []:
          print(list(ar(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(ar(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")
        elif list(au(i)) != []:
          print(list(au(i)))
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file:
            text_file.write("%s\n"%str(list(au(i))))
        
          with open(OUTPUT_REPO + "insg.txt", "a") as text_file: 
            text_file.write("\n\n")




    ##
    test = ocr_output
    test = ''.join(filter( lambda x: x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ', test ))
    test

    text_tokens = word_tokenize(test)

    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    #test = " ".join(map(str,tokens_without_sw))
    test = tokens_without_sw




    ##
    tt = [i for i in test if i != '']
    tagged = nltk.pos_tag(tt)
    tagged


    ##
    from collections import Counter

    f = []
    for i in tagged:
      if i[1] == 'NN' or i[1] == 'VBD' or i[1] == 'VBG' or i[1] == 'DT' or i[1] == 'JJ' or i[1] == 'RB' or i[1] == 'PRP' or i[1] == 'IN' or i[1]=="WP$" or i[1]=="VBZ":
        continue
      else:
        f.append(i[0])
    unique_words = set(f)
    fcount = {}
    for words in unique_words :
            fcount[words] = f.count(words)

    c=Counter(fcount)
    y = c.most_common()
    yy = [i[0] for i in y[:20]]
    yy




    ##
    word_cloud = WordCloud(collocations = False, background_color = 'black', colormap='RdYlGn').generate(" ".join(map(str,yy)))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    plt.savefig( OUTPUT_REPO + MAIN_PDF_NAME + str(page_count) + "-cloud.png" )





    ##
    def get_source(url):
          try:
            session = HTMLSession()
            response = session.get(url)
            return response

          except requests.exceptions.RequestException as e:
            print(e)
    def scrape_google(query):

        query = urllib.parse.quote_plus(query)
        response = get_source("https://www.google.co.uk/search?q=" + query)

        links = list(response.html.absolute_links)
        google_domains = ('https://www.google.', 
                          'https://google.', 
                          'https://webcache.googleusercontent.', 
                          'http://webcache.googleusercontent.', 
                          'https://policies.google.',
                          'https://support.google.',
                          'https://maps.google.')

        for url in links[:]:
            if url.startswith(google_domains):
                links.remove(url)

        return links
    def get_results(query):
        
        query = urllib.parse.quote_plus(query)
        response = get_source("https://www.google.co.uk/search?q=" + query)
        
        return response
    def parse_results(response):
        
        css_identifier_result = ".tF2Cxc"
        css_identifier_title = "h3"
        css_identifier_link = ".yuRUbf a"
        css_identifier_text = ".IsZvec"
        
        results = response.html.find(css_identifier_result)

        output = []
        
        for result in results:

            item = {
                'link': result.find(css_identifier_link, first=True).attrs['href'],
                'text': result.find(css_identifier_text, first=True).text
            }
            
            output.append(item)
            
        return output
    def google_search(query):
        response = get_results(query)
        return parse_results(response)

    query = " ".join(map(str,yy[:5]))
    results = google_search(query)
    print(results)

    with open( OUTPUT_REPO + "links.txt", "a" ) as text_file: 
      text_file.write(str(results))


from app import send_ok
send_ok()