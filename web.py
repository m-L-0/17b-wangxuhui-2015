# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
from load_CNN import *
#from work3_CNN import *


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']#<FileStorage: 'AAAE_4a4a3728-be2d-11e7-a81f-5ce0c50e583e.jpeg' ('image/jpeg')>
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('filename:',filename)
            print('file:',type(file))
            #filedir= r'../data/test/'+filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            file_url = url_for('uploaded_file',filename=filename)
            print('filename:',filename)
            img = load_CNN2(filename,model_file='../data/model.h5')
            #print('base_model:................')#img
            #img=get_picture(base_model,file_url)
            print('img:................',img)#img
            #img=123
            
            return html + '<br><img src=' + file_url + '><br><h1>'+str(img)+'</h1>'
    return html


if __name__ == '__main__':
    #base_model = load_CNN(model_file='model.h5')
    # print('load_model')
    app.run()
