from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path


# 학습 클래스 
class trainer():
    
    def __init__(self, 
        model, 
        train_dataset, 
        valid_dataset,
        
        learning_rate,
        weight_decay,
        label_smoothing,
        batch_size,
        
        device = 'cpu',
    ):
        
        
        self.device = device
        self.model = model.to(self.device)
        
        # 데이터로더 설정
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # 배치사이즈 설정
        self.batch_size = batch_size
        
        # 학습에 필요한 옵티마이저, 손실함수 구성 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    # 1 epoch 학습 및 추론 함수
    def epoch(self, loader, train=True):
        
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        total_corrects = 0.0
        
        # 일반적인 파이토치 학습 루프 구성
        for x, y in tqdm(loader, leave=False, desc='Batch'):
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            if train:
                self.optimizer.zero_grad()
            
            logits = self.model(x)
            preds = logits.argmax(dim = 1)
            
            loss = self.criterion(logits, y)
            
            if train:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_corrects += torch.sum(preds == y).item()
        
        total_loss /= len(loader.dataset)
        total_corrects /= len(loader.dataset)
            
        return total_loss, total_corrects
        
    # 학습 함수
    def fit(self, save_dir, save_name, max_epoch=30, patience=3):
        
        best_loss = float('inf')
        patience_counter = 0
        
        # 학습 기록, valid dataset 기록 
        train_history = []
        valid_history = []
        
        
        base_path = Path(save_dir)
        
        # 학습 진행, 만약 검증 손실이 특정 횟수 이상 증가 시 학습 종료 ( Early Stopping 구현 )
        bar = tqdm(range(max_epoch), desc='Epoch')
        for _ in bar:
            
            train_history.append(self.epoch(self.train_loader, train=True))
            valid_history.append(self.epoch(self.valid_loader, train=False))
            
            current_loss = valid_history[-1][0]
            
            # 중간 상황 tqdm 으로 시각화
            bar.set_postfix(
                name=save_name,
                valid_loss=f'{current_loss:.2f}',
                valid_acc=f'{valid_history[-1][1]:.2f}'
            )
            
            # 성능이 좋아졌으면 (Loss 감소)
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(self.model.state_dict(), base_path.joinpath(f'{save_name}.pth'))
                patience_counter = 0  # 카운터 초기화
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 가장 낮은 Loss 모델 재로딩
        self.model.load_state_dict(torch.load(base_path.joinpath(f'{save_name}.pth')))
        self.model.eval()
        
        # onnx 로 변환
        onnx_program = torch.onnx.export(self.model, self.train_loader.dataset[0][0].unsqueeze(0).to(self.device), dynamo=True)
        onnx_program.save(base_path.joinpath(f'{save_name}.onnx')) # type: ignore

        return train_history, valid_history

    