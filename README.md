# ImageFlow : Auto dataset collection and AutoML training pipeline for PyTorch image classification


## Installation

- 해당 프로젝트는 파이어폭스, git 이 설치되어 있어야 합니다.

```pip install git+https://github.com/naturesh/ImageFlow.git```


## Examples


##### create dataset & train model

```python
from ImageFlow.flow import Flow
from ImageFlow.report import CreateReport
from ImageFlow.data import DatasetClient

from PIL import Image
import shutil
import os


dataset_targets = [
    {
        'query' : '고양이',                                           # 구글에 검색될 단어
        'reference_images' : [Image.open('./references/고양이.png')], # 레퍼런스 이미지
        'threshold' : 0.3,                                          # 구글에서 다운로드 된 이미지 중 래퍼런스와 코사인 유사도가 threshold 이상인 경우 저장됨.
        'scroll' : 1                                                # 구글 이미지 페이지에서 스크롤 횟수 ( 클수록 많은 데이터가 수집 됨 )
    },
    {
        'query' : '사과',
        'reference_images' : [Image.open('./references/사과.png')],
        'threshold' : 0.3,
        'scroll' : 1
    },
    {
        'query' : '자동차',
        'reference_images' : [Image.open('./references/자동차.png')],
        'threshold' : 0.3,
        'scroll' : 1
    },
    {
        'query' : '포도',
        'reference_images' : [Image.open('./references/포도.png')],
        'threshold' : 0.3,
        'scroll' : 1
    },
]

# flow configs
model_type = 'tiny'
dataset_path = './outputs/dataset'
checkpoint_path = './outputs/checkpoint'
report_path = './outputs/train_report.docx'

device='mps'    # 학습 디바이스 ( mps, cuda, cpu .. )
valid_ratio=0.2 # 검증 데이터셋 비율 (0 ~ 1)

# random search params
random_params = {
    'learning_rate'  : [1e-5, 1e-4, 1e-3], # 학습률
    'weight_decay'   : [0, 1e-4, 1e-5],    # AdamW 옵티마이저 weight_decay
    'label_smoothing': [0, 0.01],          # Cross Entropy Loss 레이블 스무딩
    'batch_size'     : [16, 32],           # 배치사이즈
    'FFT'            : [False]             # 풀파인튜닝 여부 ( False : Head 만 학습 )
}
random_n = 2                               # 랜덤 서치 반복 횟수



# Checkpoint Init
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)

# Dataset Init
if os.path.exists(dataset_path) and dataset_targets:
    shutil.rmtree(dataset_path)


# create dataset
if dataset_targets:
    dc = DatasetClient(dataset_path)
    
    for i in dataset_targets: # type: ignore
        dc.run(**i)
    
    dc.browser.close()


# fit models
iflow = Flow(
    model_type,
    dataset_path,
    checkpoint_path,
    device,
    valid_ratio,
    random_params,
    random_n
)

model_train_states = iflow.fit()


CreateReport(model_train_states, report_path)
```



##### use trained model

```python
from ImageFlow.pipe import folder_classification

from ImageFlow.model import MODEL_OBJECTS



# from report.docx

onnx_path = './models/squeezenet1_1-v1.onnx' # train report 에서 제공된 경로
providers = ['CPUExecutionProvider']


transform = MODEL_OBJECTS['squeezenet1_1'].transforms() # 모델 종류에 맞게 transform 로드

class_to_idx = { # train report 에서 제공된 class 정보
    '고양이': 0, 
    '사과': 1, 
    '자동차': 2, 
    '포도': 3
}

target_folder = './target_folder'
output_folder = './outputs'


folder_classification(
    onnx_path,
    providers,
    transform,
    class_to_idx,
    target_folder,
    output_folder
)
```


##### text based classification 

```python
from ImageFlow.pipe import folder_classification_with_text

target_folder = './target_folder'
output_folder = './outputs'

folder_classification_with_text(
    texts=[
        ["a photo of duck"],
        ["a picture of a bird, not a duck"]
    ],
    target_folder=target_folder,
    output_folder=output_folder
)

```




