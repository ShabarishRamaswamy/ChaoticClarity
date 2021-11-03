from flask import Flask
from flask import request
import requests

import json

app = Flask(__name__)
import os

from dotenv import dotenv_values
config = dotenv_values(".env")

@app.route('/ping')
def ping():
    return 'Hello, World!'


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/pdf', methods=['POST'])
def post_pdf():
    return 'Got the PDF'


@app.route('/pdf', methods=['GET'])
def get_pdf():
    return 'Sent the PDF'


@app.route('/processPDF', methods=['POST'])
def process_pdf():
    all_data = request.get_data()
    all_data = json.loads(all_data)
    # print(all_data["filename"])
    # send_ok()
    os.environ["MAIN_PDF_NAME"] = all_data["filename"]
    import chaoticClarity
    chaoticClarity.sendPDFName("filename")
    # sendPDFName(all_data["filename"])
    return 'Working on it.'

def send_ok():
    requests.get("http://localhost:5000/PDFProcess/" + str(os.getenv("MAIN_PDF_NAME")))
