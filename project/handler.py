from flask import render_template, request, redirect, url_for, flash,  abort, session, jsonify , Blueprint
from werkzeug.utils import secure_filename
import os
import cv2
import urllib


class Handler:
    def __init__(request):
        self.request = request
        self.form = self.request_to_dict(request)
        from_url = 'url' in form

    def request_to_dict(self, request):
        form = self.request.args.to_dict()
        if len(form)==0:
            form = request.form.to_dict()
        return form
        
        

    def process_request_from_url(self):     
        filename = f"out_{time.time()}.png" 
        path =  form['url']   
        req = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        path = os.path.join(DIR, "static", "outputs", filename)
        img_saved_successfully = cv2.imwrite(path, img)
        return img, path, img_saved_successfully


    def process_request_from_upload(self):     

        file = self.request.files['file']
        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower()
        #extension_is_allowed = ext in ALLOWED_EXTENSIONS
        
        path = os.path.join(DIR, "static", "outputs", filename) 
        file.save(path)  
        img = cv2.imread(path)
    
        img_saved_successfully = not img is None
        return img, path, img_saved_successfully


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
