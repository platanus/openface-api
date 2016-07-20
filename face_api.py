from flask import Flask
from flask import request
import json
from flask import Response
import logging
import datetime
import face_compare

app = Flask(__name__)

@app.route('/status/')
def hello_world():
    return 'Service up on ' + str(datetime.datetime.now())

@app.route('/compare/', methods=['GET','POST'])
def login():    
    if not 'img1' in request.form:
         return "Img1 param missing", 400
    if not 'img2' in request.form:
         return "Img2 param missing", 400
    img1 = request.form['img1']
    img2 = request.form['img2']
    distance = face_compare.perform(img1, img2)
    dat = json.dumps({"distance": distance})
    resp = Response(response = dat,
    status = 200,
    mimetype = "application/json")
    return(resp)