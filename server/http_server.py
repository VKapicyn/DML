from flask import Flask, url_for, jsonify
import detect
import os

app = Flask(__name__)

@app.route('/')
def index():
    return url_for('static', filename="index.html")

@app.route('/api/v1.0/upload', methods=['POST'])
def login():
    print("заглушка загрузки картинки")
    #TODO: проверка на расширение
    #f = request.files['upload']
    #ext = f.filename.split(".")
    #if len(ext)>0:
    #    ext = ext[1]
    #else
    #    ext = "jpg"
    ext = "png"
    
    salt = "123"
    path = 'server/uploaded/'+salt+'.'+ext
    #f.save(path)

    aug = detect.load_image(path)
    os.remove(path)

    return jsonify({
        'img_url': url_for('static', filename=aug['img_name']),
        'predict_age': aug['predict_age'].tolist(),
        'predict_gender': aug['predict_gender'].tolist(),
    })
    
app.run(host='127.0.0.1', port='8000')