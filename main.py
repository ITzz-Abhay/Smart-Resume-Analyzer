import re
import nltk
from nltk.corpus import stopwords
import spacy
import pickle
import torch
from model_nn import MyNN

# importing my tfidf files and label encoding files
with open("tfidf_vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)
model=MyNN(num_features=5000)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with open("Label_encoding.pkl","rb") as f:
    encoder=pickle.load(f)

# Time for preproceesing part , it will hadle the preprocessing part
nltk.download('stopwords')


nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

def preprocess_resume(text):
        text=text.lower()

        # remove
        text=re.sub(r'http\S+|www\S+|https\S+','',text)      # remove the www and http
        text=re.sub(r'\d+','',text)                          # remove the digits 
        text=re.sub(r'\S+@\S+','',text)                      # remvoe the emails
        
        tokens=[word for word in text.split() if word not in stop_words]

        # we will now do Lemmetization
        doc=nlp(" ".join(tokens))
        tokens=[word.lemma_ for word in doc]
        return " ".join(tokens)

# model prediction starts from here
def prediction(text,model,vectorizer,encoder):
            text=preprocess_resume(text)
            features=vectorizer.transform([text]).toarray()
            features=torch.tensor(features,dtype=torch.float32)
            with torch.no_grad():
                output=model(features)
                _, predicted=torch.max(output,dim=1)
            return encoder.inverse_transform([predicted.item()])[0]


# sample_resume = "Experienced JavaScript with skills in FullStack and web developement ."
# predicted_category = prediction(sample_resume, model, vectorizer, encoder)
# print("Predicted Category:", predicted_category)

from fastapi import FastAPI

from pydantic import BaseModel

app=FastAPI()
class Resume(BaseModel):
      text:str

@app.post("/predict/")
def predict(resume:Resume):
      category=prediction(resume.text,model,vectorizer,encoder)
      return {"category":category}
      