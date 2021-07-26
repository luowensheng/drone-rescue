## set FLASK_ENV=development  (flask run -h localhost -p 3000)
import json
import numpy as np
import os.path
from flask import render_template, request, redirect, url_for, flash,  abort, session, jsonify , Blueprint
from werkzeug.utils import secure_filename
import cv2
import urllib
from .model import KRCNN
from .handler import RequestHandler
import time
from . import utils 



bp = Blueprint('project', __name__)
model = KRCNN()



@bp.route("/")
def home():   
    return render_template("home.html", contains_prediction=False)



@bp.route("/pred", methods=["POST"])
def pred()->str:

    request_handler = RequestHandler(request)
    request_result = request_handler.process_request()
    
    if request_result.failed:
        flash(request_result.message)
        return render_template("home.html", contains_prediction=False)

    path = os.path.join("static", "outputs", request_handler.filename) 

    out_paths = []
    make_prediction(request_handler, 
                    out_paths,
                    patchsize=500, 
                    do_patch_prediction=False
                    )
    n=len(out_paths)
    flash("The image has been processed succesfully! Here are the results:\n")
    return  render_template("home.html", contains_prediction=True, path=path, out_paths=out_paths, n=n)



def make_prediction(request_handler: RequestHandler, out_paths: list, patchsize:int, do_patch_prediction:bool=False) -> None:

    if do_patch_prediction:
       output_img = utils.run_on_patches(model,
                                         request_handler.img, 
                                         patchsize=patchsize,
                                         add_separation_lines = True, 
                                         width = 5
                                         )
    else:
       output_img = model(request_handler.img)
    
    out_path = os.path.join("static", "outputs", request_handler.output_filename)
    cv2.imwrite(os.path.join(request_handler.DIR, out_path), output_img )
    out_paths.append(out_path)

       

        


