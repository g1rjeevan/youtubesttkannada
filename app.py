from flask import Flask, redirect, url_for, request, render_template, session
from pytube import YouTube
import os

import requests

app = Flask(__name__, template_folder='template')

@app.route('/', methods=['GET'])
def index():
   return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
   yt = request.form['text']
   yt = YouTube(yt)
   video = yt.streams.filter(only_audio = True).first()
   destination = '.'
   out_file = video.download(output_path = destination)
   base, ext = os.path.splitext(out_file)
   new_file = base + '.mp3'
   os.rename(out_file, new_file)
   return render_template('results.html')


def audioConversion():
   return "test"