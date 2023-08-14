import glob
import os

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from utils.predictor import predict

UPLOAD_FOLDER = 'static/uploads/'

application = Flask(__name__)
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route('/')
def upload_form():
    return render_template('login.html')


@application.route('/login', methods=['POST'])  # define login page path
def login():  # define login page fucntion
    # if the request is POST the we check if the user exist
    # and with te right password
    username = request.form.get('username')
    password = request.form.get('password')

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it
    # to the hashed password in the database

    credentials = {'habibam': '3mJbFAdm',
                   'nadiadz': '9hTg5N4U',
                   'shihgianl': 'fwScBkBJ'}

    if username in credentials and credentials[username] == password:
        return render_template('upload.html')
    # if the above check passes, then we know the user has the
    # right credentials
    else:
        flash('Incorrect login credentials')
        return render_template('login.html')


@application.route('/', methods=['POST'])
def upload_image():
    # delete any files in the static/uploads directory so app memory doesn't build up
    directory = os.getcwd()
    files = glob.glob(f'{directory}/static/uploads/*')
    for f in files:
        os.remove(f)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        input_image_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)

        predicted_picture = predict(input_image_path)
        predicted_picture.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


@application.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    application.run()
