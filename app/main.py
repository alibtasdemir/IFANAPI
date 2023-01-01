from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import base64
import os
import cv2
from codecs import encode
from runtest import runTest


# Data model for handling the incoming image from the app.
class SentImage(BaseModel):
    img_enc: str


# Create api instance with FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# This function decodes the incoming base64 image to jgp format
# and save to the local
def encode_image(path, ext='jpg'):
    response_img = cv2.imread(path)
    _, response_img = cv2.imencode('.' + ext, response_img)
    response_img = base64.b64encode(response_img)
    return response_img.decode()


# Welcoming message to the API to check if the app correctly
# connected to the API
@app.get("/")
async def root():
    return {"message": "IFAN API"}


# This function receives the image sent from the app, process and
# response the processed image back to the app.
@app.post("/image")
async def image(data: Request):
    data = await data.json()
    request = dict()
    # Get image base64 from the json
    request["img_enc"] = data['_parts'][0][1]
    # delete unnecessary data
    del data
    bytes_img = encode(request["img_enc"], 'utf-8')
    img = base64.decodebytes(bytes_img)
    # Decode and save the received image
    user_image = os.path.join('input_images', 'image.jpg')
    with open(user_image, 'wb') as out:
        out.write(img)

    # Run the test function to call IFAN on the received image
    runtime = runTest()
    response_img_path = os.path.join('result_images', 'image.jpg')
    response_img = encode_image(response_img_path, ext='jpg')
    # Generate a response json with the runtime and processed image
    response_json = {
        'response': str(response_img),
        'runtime': runtime
    }
    return response_json


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=80, reload=True)
