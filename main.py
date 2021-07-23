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
from . import utils 



bp = Blueprint('project', __name__)
model = KRCNN()

DIR = os.path.join(os.getcwd(), "project")

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
        path = os.path.join(DIR, "static", "outputs", filename)
        done = cv2.imwrite(path, img)
        if not done:
            return  render_template("home.html", has_pred=False)
                    
    else:
        print(f"\n\n\n\n{'#'*20}\n\n\n")
        file = request.files['file']
        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower() 
        print(f"\n\n\n\n{'#'*20}\n\n\n")
        if not ext in ['jpeg','gif','png','jpg']:
            return  render_template("home.html", has_pred=False)
        
        fpath = os.path.join(DIR, "static", "outputs", filename) 
        file.save(fpath)  
        img = cv2.imread(fpath)

    if img is None:
       return  render_template("home.html", has_pred=False) 

    path = os.path.join("static", "outputs", filename) 

    out_paths = []
    make_prediction(img, 
                    out_paths,
                    filename=f"out_pred_{filename}", 
                    patchsize=500, 
                    with_patches=True
                    )
    n=len(out_paths)
    return  render_template("home.html", has_pred=True, path=path, out_paths=out_paths, n=n)



def make_prediction(img, out_paths, filename, patchsize, with_patches=False):

    if with_patches:
       output_img = utils.run_on_patches(model,
                                         img, 
                                         patchsize=patchsize,
                                         add_separation_lines = True, 
                                         width = 5
                                         )
    else:
       output_img = model(img)
    
    out_path = os.path.join("static", "outputs", filename)
    cv2.imwrite(os.path.join(DIR, out_path), output_img )
    out_paths.append(out_path)

       

        


