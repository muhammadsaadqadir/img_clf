# import io
# import pickle
# import numpy as np
# import PIL.Image
# import PIL.ImageOps
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(CORSMiddleware, 
#                    allow_origins=["*"],
#                    allow_credentials=True,
#                    allow_methods=["*"],
#                    allow_headers=["*"])

# # Load the model when the application starts
# @app.on_event("startup")
# def load_model():
#     global model
#     with open('mnist_cnn_model.pkl', 'rb') as f:
#         model = pickle.load(f)

# @app.post("/predict-image/")
# async def predict_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
#     pil_image = PIL.ImageOps.invert(pil_image)
#     pil_image = pil_image.resize((28, 28), PIL.Image.BICUBIC)
#     img_array = np.array(pil_image).reshape(1, -1)
#     prediction = model.predict(img_array)

#     return {"prediction": int(prediction[0])}







import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# Global variable to hold the loaded CNN model
model = None

# Load the CNN model when the application starts
@app.on_event("startup")
def load_model():
    global model
    with open('mnist_cnn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Assuming you saved the entire model including its architecture, optimizer, and weights

# Endpoint to predict the image
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), PIL.Image.BICUBIC)
    img_array = np.array(pil_image).reshape(1, 28, 28, 1)  # Reshape to match CNN input shape
    prediction = model.predict(img_array)

    return {"prediction": int(np.argmax(prediction))}  # Assuming prediction is one-hot encoded, convert to label index

