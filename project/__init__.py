from flask import Flask



def create_app(test_config=None):
    app = Flask(__name__)
    #app.secret_key = 'h432hi5ohi3h5i5hi3o2hi'
    app.config['SECRET_KEY'] = 'h432hi5ohi3h5i5hi3o2hi'


    from . import entry 

    app.register_blueprint(entry.bp)
    return app