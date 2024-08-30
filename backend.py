from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()

# CORS configuration
origins = [
    "http://127.0.0.1:5500",  # Update this to match your frontend's URL
    "http://localhost:5500",  # You can add additional origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model(r'E:\crop-disease-prediction\model.h5')

def preprocess_image(image_bytes):
    img = keras_image.load_img(io.BytesIO(image_bytes), target_size=(150, 150))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    return img_array / 255.0  # Normalization

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array)
    class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
    predicted_class = class_names[np.argmax(predictions)]
    return JSONResponse(content={"prediction": predicted_class})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
