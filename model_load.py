"""
Trains an Alternating Least Squares (ALS) model for user/movie ratings.
The input is a Parquet ratings dataset (see etl_data.py), and we output
an mlflow artifact called 'als-model'.
"""

import click
import mlflow
import mlflow.keras
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import json
import ast
app = flask.Flask(__name__)
def load_model(model_dir):
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    print("load_model model_dir0",model_dir )
    model_dir=model_dir.replace('\\','/')
    print("load_model model_dir1",model_dir )
    model_dir=r'file:///' + model_dir
    print("load_model model_dir",model_dir )
    model =mlflow.keras.load_model(model_dir[:-1])
#    model =mlflow.keras.load_model(r'file:///C:/Users/AmitCahanovich/Documents/personal/versatile/mlflow/examples/multistep_workflow/mlruns/0/982847f8c6a74029bd8f2a718f6ec463/artifacts/keras-model')
#    model =mlflow.keras.load_model('../mlflow/examples/multistep_workflow/mlruns/0/0ccdde176f1447a29d2be2f4dfcec583/artifacts/keras-model')
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(150, 150))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            print("predictions %s" % preds[0])
            print("argmax",np.argmax(preds[0]))
            argmax=np.argmax(preds[0])
#            labels=['claude_monet', 'frida_kahlo', 'jackson_pollock', 'jose_clemente_orozco', 'salvador_dali', 'vincent_van_goh']
 
            data['confidence']=str(int(100 *preds[0][argmax]) ) + '%'
#            data['prediction']=labels[argmax]
            data['prediction']=label_json_dict[argmax]
            data['success']=True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)
@click.command()
@click.option("--model-dir", help="model dir")
@click.option("--label-dir", help="label file")

def model_inference(model_dir, label_dir):
    print ('model dir:%s' % model_dir)
    load_model(model_dir )  
    print ('label_dir:%s' % label_dir)
    global label_json_dict
    with open(label_dir[1:-1], 'r') as file:
         labels=file.read()
    label_json_dict=ast.literal_eval(labels[0:-1])
    print(label_json_dict)
      
    app.run()



if __name__ == "__main__":
    model_inference()
