import os
from fastapi import FastAPI, File, UploadFile
import requests
import io
import uvicorn
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google.cloud import storage
from app.utils.utilsmain import get_image_from_bytes
import imageio
import os
from app.make_video import gradio_funct

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "app/google_auth.json"
# Instantiates a client
client = storage.Client()
bucket_id = "wombo_bucket"
bucket = client.get_bucket(bucket_id)

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""return black and white pic""",
    version="0.0.1",
)


origins = ["http://localhost", "http://localhost:80:80", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def main():
    """Implement a get method that returns always the same video for test purpose"""
    return {"url": "https://storage.googleapis.com/wombo_bucket/driving.mp4"}


@app.post("/img_to_img/")
async def bandw_im_im(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file))  # .convert("RGB")
    # input_image = get_image_from_bytes(file)
    img_pil = input_image.convert("1")
    bytes_io = io.BytesIO()
    # img_base64 = Image.fromarray(img_pil)
    img_pil.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


@app.post("/img_to_url/")
async def bandw_im_url(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file))  # .convert("RGB")
    # input_image = imageio.imread(io.BytesIO(file))
    input_image = get_image_from_bytes(file)
    print(input_image)
    img_pil = input_image.convert("1")
    # save in google storage
    img_pil.save("tmp_im.jpg")
    new_blob = bucket.blob("img_test.jpg")
    new_blob.upload_from_filename(filename="tmp_im.jpg")
    new_blob.make_public()
    public_url = new_blob.public_url
    return {"url": public_url}


@app.post("/url_to_url/")
async def bandw_url_url(url: str):
    if url[0:6] != "https:":
        url = "https:" + url
    image = requests.get(url).content
    source_image = imageio.imread(image)
    created_vid = gradio_funct(source_image, video_num=1)

    # Save in Google bucket
    new_blob = bucket.blob(created_vid.split("/")[-1])
    new_blob.upload_from_filename(filename=created_vid)

    # to make the url public
    new_blob.make_public()
    public_url = new_blob.public_url
    return {"url": public_url}
