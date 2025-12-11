from torchvision.datasets import ImageFolder
from typing import Literal
from torch.utils.data import random_split

from .model import model_cards, trainer

import random
import os

import torch
import gc

class Flow():
    
    def __init__(self, 
        
        # 모델 타입 tiny는 초소형 모델, large는 대형 모델
        model_type : Literal['tiny', 'small', 'base', 'large'], 
        dataset_path, # 만들어진 데이터셋 경로
        checkpoint_path : str, # 체크포인트를 저장할 경로
        
        device, # mps, cuda, cpu 등 
        valid_ratio, # validation set으로 얼마나 쓸지 (0~1) 비율
        
        # random search params 랜덤 서치 파라미터 목록
        random_params = {
            'learning_rate'  : [1e-5, 1e-4, 1e-3],
            'weight_decay'   : [0, 1e-4, 1e-5],
            'label_smoothing': [0, 0.01],
            'batch_size'     : [16, 32],
            'FFT'            : [False]
        },
        random_n = 3 # 각 모델마다 몇번 랜덤 반복할지
        
    ):
        
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
                
        self.models = model_cards[model_type]
        
        self.device = device
        self.valid_ratio = valid_ratio
    
        self.random_params = random_params
        self.random_n = random_n
    
    def fit(self):
        
        os.makedirs(self.checkpoint_path, exist_ok=True)

        states = [] # 모델별 학습 로그 및 파라메터 저장 리스트
        dataset = ImageFolder(self.dataset_path, transform=None)
        
        # train, valid 분리 
        valid_size = int(len(dataset) * self.valid_ratio)
        train_dataset, valid_dataset = random_split(dataset, [len(dataset)-valid_size, valid_size])
        
        for model_builder, obj in self.models:
            
            # 데이터셋 객체 transform 을 각 모델의 transform 으로 변경 (tramsform : 전처리 함수)
            train_dataset.dataset.transfrom = obj.transforms() # type: ignore
            valid_dataset.dataset.transform = obj.transforms() # type: ignore
            
            for n in range(self.random_n):
                
                # 모델 로드, 헤드 수는 데이터셋의 클래스 수로 설정
                model = model_builder(len(dataset.classes), obj.get_state_dict())
                               
                state = {
                    
                    'n'            : n, # 몇번째 반복 모델인지 (버전)
                    'type'         : model_builder.__name__, # 모델 이름
                    'class_to_idx' : dataset.class_to_idx, # 학습한 데이터 클래스
                    
                    'save_dir' : self.checkpoint_path, # 저장 경로
                    'save_name': f'{model_builder.__name__}-v{n}' # 저장 이름
                    
                }
                
                state['params'] = { k : random.choice(v) for k, v in self.random_params.items() } # 랜덤 서치 파라미터 선정

                if state['params']['FFT']: # 풀 파인 튜닝 파라미터가 True 라면, 모델 전체 학습 가능하도록. (기존에는 헤드만 학습) 
                    for param in model.parameters():
                        param.requires_grad = True
                
                # 모델 학습 객체 생성
                t = trainer(
                    model=model,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    
                    learning_rate=state['params']['learning_rate'],
                    weight_decay=state['params']['weight_decay'],
                    label_smoothing=state['params']['label_smoothing'],
                    batch_size=state['params']['batch_size'],
                    
                    device=self.device,
                )
                
                # 모델 학습 및 학습, 검증 기록 저장
                state['train_history'], state['valid_history'] = t.fit(
                    save_dir=state['save_dir'],
                    save_name=state['save_name']
                )
                
                # 검증 손실이 가장 낮을 때 검증 손실, 정확도 저장 
                state['valid_loss'], state['valid_acc'] = min(state['valid_history'], key=lambda x: x[0])
                
                # 모델 학습 기록 
                states.append(state)
                
                
                # clear memory 메모리 정리
                del t
                
                gc.collect()
                
                # 그래픽 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.mps.is_available():
                    torch.mps.empty_cache()
                
                    
        return states
               
