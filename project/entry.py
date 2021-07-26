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

ALLOWED_EXTENSIONS  =   ['jpeg','gif','png','jpg']


@bp.route("/")
def home():   
    return  render_template("home.html", contains_prediction=False)



@bp.route("/pred", methods=["GET","POST"])
def pred():

    form = request.args.to_dict()
    if len(form)==0:
        form = request.form.to_dict()
    
    from_url = 'url' in form

    if from_url :   
        filename = f"out_{time.time()}.png" 
        path =  form['url']   
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        path = os.path.join(DIR, "static", "outputs", filename)
        img_saved_successfully = cv2.imwrite(path, img)

        if not img_saved_successfully:
            flash("Image has not been loaded succesfully. There might be a problem with the url!")
            return  render_template("home.html", contains_prediction=False)
                    
    else:
        file = request.files['file']
        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower()
        extension_is_allowed = ext in ALLOWED_EXTENSIONS

        if not extension_is_allowed:
            flash(f"Extension [{ext}] is not allowed!\nHere is a list of allowed extenssions:\n\t{ALLOWED_EXTENSIONS}")
            return  render_template("home.html", contains_prediction=False)
        
        fpath = os.path.join(DIR, "static", "outputs", filename) 
        file.save(fpath)  
        img = cv2.imread(fpath)
    
    img_not_read = img is None

    if img_not_read:
       flash("Image not loaded correctly. Please try again.")
       return  render_template("home.html", contains_prediction=False) 

    path = os.path.join("static", "outputs", filename) 

    out_paths = []
    make_prediction(img, 
                    out_paths,
                    filename=f"out_pred_{filename}", 
                    patchsize=500, 
                    do_patch_prediction=False
                    )
    n=len(out_paths)
    flash("Image has been processed succesfully! Here are the results:\n")
    return  render_template("home.html", contains_prediction=True, path=path, out_paths=out_paths, n=n)



def make_prediction(img: np.array, out_paths: list, filename:str, patchsize:int, do_patch_prediction:bool=False) -> None:

    if do_patch_prediction:
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

       

        


