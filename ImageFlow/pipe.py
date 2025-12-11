import onnxruntime
import numpy as np

from .data import ImageSimilarity
from PIL import Image

from pathlib import Path
from tqdm import tqdm


# 학습된 모델을 손쉽게 사용할수 있도록 하는 클래스
class Pipe():
    
    def __init__(self, onnx_path, providers, transform, class_to_idx):
        
        self.model = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.transform = transform
        self.idx_to_class = {v:k for k,v in class_to_idx.items()}
    
    def __call__(self, img: Image.Image):
        
        # 모델 인풋값 설정
        inputs = {
            self.model.get_inputs()[0].name: self.transform(img).unsqueeze(0).cpu().numpy()
        }
        # 모델 출력 구하기
        outputs = self.model.run(None, inputs)[0]

        # idx에 맞는 클래스로 출력
        return self.idx_to_class[np.argmax(outputs)] # type: ignore
        
# 학습된 모델을 기반으로 target 폴더를 분류하는 함수
def folder_classification(onnx_path, providers, transform, class_to_idx, target_folder, output_folder):
    
    p = Pipe( # 파이프라인 생성
        onnx_path,
        providers,
        transform,
        class_to_idx
    )
    
    target_folder = Path(target_folder) # 분류할 이미지가 담겨있는 경로
    output_folder = Path(output_folder) # 분류된 이미지가 저장될 경로
    
    # 분류된 이미지가 저장될 폴더 생성
    for c in class_to_idx.keys():
        output_folder.joinpath(c).mkdir(parents=True, exist_ok=True)
    
    # 해당 확장자만 분류
    allow_ext = ['.jpg', '.png', '.jpeg']
    
    
    
    for target in tqdm(target_folder.glob('*'), desc='Folder Classification'):
        
        # allow_ext 에 해당하는 확장자라면
        if target.suffix.lower() in allow_ext:
            img = Image.open(target) 
            
            # 이미지 분류
            cls = p(img)
            
            # 이미지 저장
            img.save(output_folder.joinpath(cls, target.name))
        

# 텍스트 기반 이미지 분류, 모델 학습만으로 분류가 어려운 경우, 및 학습 없이 실행이 필요한 경우
def folder_classification_with_text(texts, target_folder, output_folder):
    
    target_folder = Path(target_folder)
    output_folder = Path(output_folder)
     
    # 분류된 이미지 저장할 폴더 이름 구성
    dirs = [i[0].replace(' ', '_') for i in texts]
     
    # 폴더 생성
    for d in dirs:
        output_folder.joinpath(d).mkdir(parents=True, exist_ok=True)
    
    allow_ext = ['.jpg', '.png', '.jpeg']
    
    # 이미지 유사도 체크 클래스 생성, 기본 ref_img, ref_text 미사용
    sim = ImageSimilarity(ref_img=[], ref_text=[])
    
    # 분류할 각 기준 텍스트 마다 대표 임베딩 생성
    embeds = [sim.reference_embed(imgs=[], texts=i) for i in texts]
    
    for target in tqdm(target_folder.glob('*'), desc='Folder Classification'):
        
        if target.suffix.lower() in allow_ext:
            
            img = Image.open(target) 
            
            # 가장 유사도가 높은 기준 텍스트 찾기 및 해당 폴더 저장 
            best_sim = -float('inf')
            dir_name = ''
            
            for idx, e in enumerate(embeds):
                
                similarity = sim.check(img, e)
                if similarity > best_sim:
                    dir_name = dirs[idx]
                    best_sim = similarity
                    
            # 이미지 저장
            img.save(output_folder.joinpath(dir_name, target.name))
            