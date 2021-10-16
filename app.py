from flask import Flask
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
