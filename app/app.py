import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predictor import load_model, load_and_preprocess_image, get_prediction

model = load_model()

app = Flask(__name__)


@app.route('/')
def hello_world() -> str:
    return 'Hello World!'


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route("/image", methods=["GET", "POST"])
def get_image():
    if request.method == 'POST':
        f = request.files['file']
        sfname = os.path.join('static/images/', str(secure_filename(f.filename)).lower())
        f.save(sfname)
        image = load_and_preprocess_image(sfname, path='')
        predictions = get_prediction(image, model)
        prediction_str = "This cat most likely is a {} with a {:.2f}" \
                         " percent confidence.".format(predictions[0][np.argmax(predictions[1])],
                                                       100 * np.max(predictions[1]))

        return render_template('prediction.html', pred_data=zip(*predictions), prediction=prediction_str,
                               image='../' + sfname)


if __name__ == '__main__':
    app.run()
