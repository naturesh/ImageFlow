from PIL import Image

import open_clip
import torch

# 이미지 및 텍스트 유사도 검증을 도와주는 클래스
class ImageSimilarity():
    
    def __init__(self, ref_img, ref_text):
        
        # 모델 로드
        self.model, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()
        
        # 토크나이저 로드
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # 참고할 임베딩 생성 
        self.reference = self.reference_embed(ref_img, ref_text)
        
    # 임베딩 생성 함수 
    def encode(self, data):

        with torch.no_grad():
            # 이미지 객체라면 encode_image 사용
            if isinstance(data, Image.Image):
                return self.model.encode_image(self.transform(data).unsqueeze(0)) # type: ignore
            
            # 문자열이라면 encode_text 사용
            elif isinstance(data, str):
                return self.model.encode_text(self.tokenizer([data])) # type: ignore
        
    
    # 정규화된 참고 임베딩 생성 
    def reference_embed(self, imgs, texts):
        
        ref = [self.encode(i) for i in imgs+texts]
        
        if ref:
            ref = torch.cat(ref).mean(dim=0, keepdim=True) # type: ignore
            ref/= ref.norm(dim=-1, keepdim=True)
            
            return ref.T
        else:
            return None
    
    # 유사도 체크 
    def check(self, img, target=None):
        
        # 이미지 임베딩 생성 및 정규화
        encoded = self.encode(img)
        encoded/= encoded.norm(dim=-1, keepdim=True) # type: ignore
        
        # cosine similarity 코사인 유사도 체크 (target이 존재한다면 target 과 아니라면 기본 참고 임베딩과)
        if target is not None:
            return (encoded @ target).item()
        else:
            return (encoded @ self.reference).item()
        
    
    

