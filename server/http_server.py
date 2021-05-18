from flask import Flask, url_for, jsonify, request
from random import randint
from flask_cors import CORS
import detect
import os

app = Flask(__name__, static_url_path='/static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/api/v1.0/upload', methods=['POST'])
def login():

    f = request.files['upload']
    print(f.filename)
    ext = f.filename.split(".")
    if len(ext)>0:
        ext = ext[len(ext)-1]
    else:
        ext = "jpg"
    
    salt = str(randint(1, 99999999))
    path = 'server/uploaded/'+salt+'.'+ext
    f.save(path)

    aug = detect.load_image(path)
    os.remove(path)

    if 'error' in aug:
        return jsonify({
            'error': 'Лица не найдены'
        })
    else:
        return jsonify({
            'img_url': url_for('static', filename=aug['img_name']),
            'predict_age': aug['predict_age'].tolist(),
            'predict_gender': aug['predict_gender'].tolist(),
        })
    
app.run(host='0.0.0.0', port='8000')