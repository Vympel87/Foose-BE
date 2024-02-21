from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import keras_ocr
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from openfoodfacts import API  # Correct import
from datetime import datetime, timedelta

app = FastAPI()

ocr_pipeline = keras_ocr.pipeline.Pipeline()
api = API(version="v2")

def read_image(url: str) -> np.ndarray:
    response = requests.get(url)
    image = np.array(Image.open(BytesIO(response.content)))
    return image

def get_food(product_name: str) -> dict:
    try:
        products = api.product.text_search(product_name)
        print(products)  # Tambahkan pernyataan cetak untuk melihat hasil dari pencarian

        expiration_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        
        if 'products' in products and products['products']:
            # Ambil produk pertama (atau pertama yang ditemukan)
            product = products['products'][0]
            
            # Ekstrak informasi yang diinginkan
            categories = product.get("categories_tags", ["Unknown"])
            category = categories[0] if categories else "Unknown"
            
            return {'expired': expiration_date, 'category': category}
        
        return {'expired': expiration_date, 'category': "Unknown"}
    except Exception as e:
        return {'error': str(e)}

@app.post("/predict/")
async def predict(image_url: str = Query(..., description="URL of the image to be processed")):
    try:
        image = read_image(image_url)
        
        predictions = ocr_pipeline.recognize([image])
        result_texts = [text for text, _ in predictions[0]]

        results = []
        for product_name in result_texts:
            result = get_food(product_name)
            results.append(result)

        return JSONResponse(content={'results': results}, status_code=200)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

@app.get("/")
async def test():
    return "Server is running"
