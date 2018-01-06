from flask import Flask, render_template, request, redirect, url_for
from use import (get_nn_result, get_svm_result)
from werkzeug.utils import secure_filename
import os
import hashlib
import threading
import queue
import time
import json


ALLOWED_EXTENSIONS = set(['mp3', 'wav', 'flac'])
predTasks = queue.Queue()
app = Flask(__name__)


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/about')
def about():
  return render_template('about.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_img_url(artist):
    if "Taylor" in artist:
        url = '/static/img/taylor_swift.jpg'
    elif "Michael" in artist:
        url = '/static/img/michael_jackson.jpg'
    elif "Ed" in artist:
        url = '/static/img/ed_sheeran.jpg'
    elif "Ariana" in artist:
        url = '/static/img/ariana_grande.jpg'
    return url


def evaluateThread():
    while True:
        if not predTasks.empty():
            filepath = predTasks.get()

            #Neural Network
            pred_nn, stats = get_nn_result(filepath)
            img_nn = get_img_url(pred_nn)
            stat_taylor = stats[0]
            stat_michael = stats[1]
            stat_ed = stats[2]
            stat_ariana = stats[3]

            #svm
            pred_svm = get_svm_result(filepath)
            img_svm = get_img_url(pred_svm)
            content = {
                'neural_network': {
                    'name': pred_nn,
                    'img': img_nn,
                    's_taylor': stat_taylor,
                    's_michael': stat_michael,
                    's_ed': stat_ed,
                    's_ariana': stat_ariana
                },
                'svm': {
                    'name': pred_svm,
                    'img': img_svm
                }
            }
            with open(filepath + '.json', 'w') as f:
                json.dump(content, f)
            os.remove(filepath)
        time.sleep(10)


@app.route('/delete', methods=['GET'])
def delete_res():
    filename = request.args.get('hash') + '.json'
    if os.path.isfile('./uploads/' + filename):
        os.remove('./uploads/' + filename)
    return redirect('/')

@app.route('/result', methods=['GET'])
def result():
    filename = request.args.get('hash') + '.json'
    if os.path.isfile('./uploads/' + filename):
        with open('./uploads/' + filename, 'r') as f:
            data = json.load(f)
        return render_template('result.html',
                                hash= request.args.get('hash'),
                                pred_nn= data['neural_network']['name'],
                                artist_url_nn = data['neural_network']['img'],
                                s_taylor= data['neural_network']['s_taylor'],
                                s_michael = data['neural_network']['s_michael'],
                                s_ed = data['neural_network']['s_ed'],
                                s_ariana = data['neural_network']['s_ariana'],
                                pred_svm = data['svm']['name'],
                                artist_url_svm = data['svm']['img'],
                                is_done = True)
    return render_template('result.html', 
                            is_done = False)
                        


@app.route('/predict', methods=['GET', 'POST'])
def predict(): 
    if 'file' not in request.files:
        return render_template('home.html', is_error=True, err_msg='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('home.html', is_error=True, err_msg='No file selected')
    if file and allowed_file(file.filename):
        filename = hashlib.sha256(secure_filename(file.filename).encode('utf-8'))
        filename = filename.hexdigest()
        file.save('./uploads/' + filename)
        predTasks.put('./uploads/' + filename)
        return redirect('/result?hash=' + filename)
    else:
        return render_template('home.html', is_error=True, err_msg='File extension not allowed')
    


if __name__ == '__main__':
    t = threading.Thread(target=evaluateThread)
    t.start()
    app.secret_key = 'some secret key'
    app.run(debug=True)
    
