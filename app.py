from flask import Flask
from flask import request
import requests

import json

from chaoticClarity import sendPDFName

app = Flask(__name__)
import os

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
    print(all_data)
    send_ok()
    sendPDFName(all_data) # Fix this to the correct value.
    return 'Working on it.'

def send_ok():
    requests.get('http://localhost:5000/processPDF/completed')
