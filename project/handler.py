from flask import render_template, request, redirect, url_for, flash,  abort, session, jsonify , Blueprint
from werkzeug.utils import secure_filename
import os
import cv2
import urllib
import time
import numpy as np


class Results:
    def __init__(self, message: str, failed:str):
        self.message = message
        self.failed = failed


class RequestHandler:

    ALLOWED_EXTENSIONS = ['jpeg','gif','png','jpg']

    def __init__(self,request):

        self.request = request
        self.form = self.request_to_dict(request)
        self.request_is_from_url = 'url' in self.form
        self.filename = None
        self.img = None
        self.path = None
        self.DIR = os.path.join(os.getcwd(), "project")

    def request_to_dict(self, request):
        form = self.request.args.to_dict()
        if len(form)==0:
            form = request.form.to_dict()
        return form
        
    def process_request(self):
        if self.request_is_from_url:
           return self._process_request_from_url()
        return self._process_request_from_upload()        

    def _process_request_from_url(self)-> Results:     

        self.filename = f"out_{time.time()}.jpg" 
        path =  self.form['url'] 
        try: 
          req = urllib.request.urlopen(path)
        except Exception as message:
           failed = True  
           return Results(message=message, failed=failed) 

        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        self.img = cv2.imdecode(arr, -1)
        self.path = os.path.join(self.DIR, "static", "outputs", self.filename)
        img_saved_successfully = cv2.imwrite(path, self.img)

        if not img_saved_successfully:
           message =  "Image has not been loaded succesfully. There might be a problem with the url!"  
           failed = True         
           
        else:
            message = None
            failed = False

        return Results(message=message, failed=failed)    

    def _process_request_from_upload(self)-> Results:  

        file = self.request.files['file']
        self.filename = secure_filename(file.filename)
        ext = self.filename.split(".")[-1].lower()
        extension_is_allowed = ext in self.ALLOWED_EXTENSIONS

        if not extension_is_allowed:
            message =  f"Extension [{ext}] is not allowed!\nHere is a list of allowed extenssions:\n\t{ALLOWED_EXTENSIONS}"
            failed = True  
        else:         
        
            self.path = os.path.join(self.DIR, "static", "outputs", self.filename) 
            file.save(self.path)  
            self.img = cv2.imread(self.path)
        
            img_not_read =  self.img is None

            if img_not_read:
               message =  f"Extension [{ext}] is not allowed!\nHere is a list of allowed extenssions:\n\t{ALLOWED_EXTENSIONS}"
               failed = True 

            else:
               message = None
               failed = False     

        return Results(message=message, failed=failed)

    @property
    def output_filename(self):

        if self.filename is None:
           return None

        return f"out_pred_{self.filename}"     


