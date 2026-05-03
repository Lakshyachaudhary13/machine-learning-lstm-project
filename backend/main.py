from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load global variables
model = None
tokenizer = None
max_sequence_len = 0

@app.on_event("startup")
async def load_assets():
    global model, tokenizer, max_sequence_len
    try:
        model = tf.keras.models.load_model('sentence_completion_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('config.pkl', 'rb') as f:
            config = pickle.load(f)
            max_sequence_len = config['max_sequence_len']
        print("Backend assets loaded successfully!")
    except Exception as e:
        print(f"Error loading assets: {e}")

class TextRequest(BaseModel):
    text: str
    top_n: int = 5

@app.get("/")
def read_root():
    return {"message": "Sentence Completion API is running"}

@app.post("/complete")
async def complete_sentence(request: TextRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model or Tokenizer not loaded")
    
    seed_text = request.text
    if not seed_text:
        return {"predictions": []}

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    top_n_indexes = np.argsort(predicted[0])[::-1][:request.top_n]
    
    predictions = []
    for index in top_n_indexes:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                predictions.append({
                    "word": word,
                    "probability": float(predicted[0][index])
                })
                break
    
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
