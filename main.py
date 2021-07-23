## set FLASK_ENV=development  (flask run -h localhost -p 3000)
import json
import numpy as np
import os.path
from flask import render_template, request, redirect, url_for, flash,  abort, session, jsonify , Blueprint
from werkzeug.utils import secure_filename
import cv2
import urllib
from .model import KRCNN
import time
from .utils import run_on_patches



bp = Blueprint('project', __name__)
model = KRCNN()



@bp.route("/")
def home():   
    return  render_template("home.html", has_pred=False)



@bp.route("/pred", methods=["GET","POST"])
def pred():

    form = request.args.to_dict()
    if len(form)==0:
        form = request.form.to_dict()
 
    if 'url' in form:   
        filename = f"out_{time.time()}.png" 
        path =  form['url']   
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        path = os.path.join(model.DIR, "static", "outputs", filename)
        done = cv2.imwrite(path, img)
        if not done:
            return  render_template("home.html", has_pred=False)
                    
    else:

        file = request.files['file']
        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower() 

        if not ext in ['jpeg','gif','png','jpg']:
            return  render_template("home.html", has_pred=False)
        
        fpath = os.path.join(model.DIR, "static", "outputs", filename) #f"out{model.NUM_IMAGES}.{ext}")  
        file.save(fpath)  
        img = cv2.imread(fpath)

    if img is None:
       return  render_template("home.html", has_pred=False) 

    path = os.path.join("static", "outputs", filename) 
    out_paths = []
    pred_path = model(img, name=f"out_pred{model.NUM_IMAGES}.png")

    return  render_template("home.html", path=path, out_paths=out_paths)

def make_prediction(img, with_patches=False):

    if with_patches:
       output_img = model(img)
    else:
       output_img = model(img) 
