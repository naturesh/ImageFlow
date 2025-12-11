from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from pathlib import Path

from .utils import src_to_base64, base64_to_img
from .similarity import ImageSimilarity
from tqdm import tqdm

import time

import gc


# 셀레니움 연결 클라이언트 
class DatasetClient():
    
    def __init__(self, save_path):
        
        # 브라우저(Firefox)를 헤드리스 모드로 구성합니다.
        service = Service()
        options = Options()
        options.add_argument('--headless')
        self.browser = webdriver.Firefox(service=service, options=options)
        
        # 데이터 저장할 경로 
        self.save_path = Path(save_path)
        
        
    def run(self, query, scroll = -1, reference_images = [], reference_texts = [], threshold = 0.4):
        
        # 구글 이미지 검색
        self.browser.get(f'https://www.google.com/search?q={query}&source=hp&sclient=img&udm=2')
        
        prev = 0 # 스크롤 전 페이지 이미지 개수 
        curr = [] # 현재 페이지에서 이미지 객체 리스트
        
        # scroll 만큼 스크롤 반복
        for _ in range(scroll if scroll > 0 else 10):
            
            # 맨 아래로 스크롤
            self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            
            # 이미지 로딩 대기
            time.sleep(3)
            
            # 현재 페이지 html에서 이미지 src 추출
            curr = self.parse(self.browser.page_source)
            
            # 스크롤 전 이미지 개수와 현재 이미지 개수가 같다면 종료 (끝까지 로딩 완료)
            if prev == len(curr): 
                break
            
            prev = len(curr)
        
        # 데이터셋 폴더 생성
        save_dir = self.save_path.joinpath(query)
        save_dir.mkdir(parents=True, exist_ok=False)
        
        # 이미지 유사도 검색 객체 생성, 수집에 참고할 이미지, 텍스트 등록 
        sim = ImageSimilarity(reference_images, reference_texts)

        
        for i, img in enumerate(tqdm(curr, desc=query)):
            try:
                # 이미지 url 주소를 base64 텍스트 변환, 이후 pillow 이미지 객체 변환
                img = base64_to_img(src_to_base64(img.get('src')))
                
                # 이미지 유사도 체크 및 저장
                similarity = sim.check(img)
                if similarity > threshold:
                    img.save(save_dir.joinpath(f'{i}.png'))
                
            except Exception as e: 
                print(e)
        
        # 메모리 정리 
        del sim
        del curr
        gc.collect()
    
    # 구글 이미지 결과 html에서 이미지에 해당하는 객체 찾기 
    def parse(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return soup.find_all(lambda tag: tag.has_attr('class') and tag.get('class') == ['YQ4gaf'])


