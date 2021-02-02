from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap 
#from flask_uploads import UploadSet, configure_uploads
#from werkzeug.utils import secure_filename
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
#import csv
#import tablib
import pandas as pd

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENTIONS = set(['csv'])
folder = "uploads/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

            
def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

#app = Flask (__name__)
#Bootstrap (app)
#model = pickle.load(open("data/indihome.pkl","rb"))
#dataset = tablib.Dataset()
#with open (os.path.join(os.path.dirname(__file__), "indihome_dataset_bersih.csv")) as f:
#    dataset.csv = f.read()

#@app.route ("/predict", methods=["POST"])
#def predict():
#    '''
#    Rendering result on HTML GUI
#    
#    '''

@app.route('/')
def index():
     return render_template ("pie.html")

@app.route('/hasil', methods = ["GET","POST"])

def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        df = pd.read_csv(file)
        df_input = df['tweet']
        vec_file = 'vectorizer.pkl'
        # Load from file
        with open(vec_file , 'rb') as file:
            pickle_vect = pickle.load(file)
       # vect= vectorizer.transform(df.values.astype('U'))
        vect = pickle_vect.transform(df_input).toarray()
        input = df.iloc[:]
        df = pd.DataFrame(input)
        pkl_filename ="coba_model_svm.pkl"
        with open(pkl_filename,'rb') as file:
            pickle_model = pickle.load(file)
      
        my_prediction =pickle_model.predict(vect)
        df['label'] = my_prediction
        df['label'] = df['label'].replace({0:'negatif',1:'positif'})
        
        data_pos = df.loc[df["label"]== 'positif']
        data_neg = df.loc[df["label"]== 'negatif']
        z= len(df)
        x= len(data_pos)
        y= len(data_neg)
        x1= x/z*100
        x2 = y/z*100
        x3 = '{:.1f}'.format(x1)
        x4 = '{:.1f}'.format(x2)
        
        labels= ['Positif', 'Negatif']
        values = [x3,x4] #nilai masing masing kategori
        colors = ["#8080ff","#d5d5c3"]
        
        return render_template('hasil.html', max=17000, title="Prediksi Analisis Sentimen Indihome", set=zip(values, labels, colors), x3=(x3),x4=(x4), z=(z), x=(x),y=(y))
        
        
    
#        df.to_csv(r'uploads\result.csv')
        #file.save (os.path.join("uploads",file.filename))
        
#        label=[]
#        return render_template ("hasil.html")
#    return render_template("hasil.html", message="Upload")
#    
##        df.to_csv(r'uploads\result.csv')
#        #file.save (os.path.join("uploads",file.filename))

if __name__ == '__main__':
#    HOST = os.environ.get("SERVER_HOST","localhost")
#    try:
#        PORT = int (os.environ.get("SERVER_PORT", "55555"))
#    except ValueError:
#        PORT = 5555
    app.run()
