from PIL import Image
from io import BytesIO

import requests
import base64


# src(url) 이미지를 요청해 base64로 변환하는 함수
def src_to_base64(src):
    
    if src.startswith('data:image'):
        return src
    
    res = requests.get(src, headers={'User-Agent': 'Mozilla/5.0'})
    return base64.b64encode(res.content).decode('utf-8')
    
# base64(이미지)를 Pillow Image로 변환하는 함수
def base64_to_img(base):
    
    if ',' in base: 
        base = base.split(',')[1]
        
    return Image.open(BytesIO(base64.b64decode(base)))
    