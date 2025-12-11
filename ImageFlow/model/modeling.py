import torchvision.models as models
import torch.nn as nn

import re

# 모델 프리징 함수
def freeze(m):
    for p in m.parameters(): 
        p.requires_grad = False
        
        
### 기본 모델 Configuration 은 그대로 사용하되 num_classes만 변경하는 모델별 Wrapper 

def squeezenet1_1(n, weight = None):
    m = models.squeezenet1_1()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[1] = nn.Conv2d(512, n, kernel_size=(1, 1), stride=(1, 1))
    return m
    
def shufflenet_v2_x1_0(n, weight = None):
    m = models.shufflenet_v2_x1_0()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.fc = nn.Linear(1024, n)
    return m

def mobilenet_v3_small(n, weight = None):
    m = models.mobilenet_v3_small()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[3] = nn.Linear(1024, n)
    return m

def mobilenet_v3_large(n, weight = None):
    m = models.mobilenet_v3_large()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[3] = nn.Linear(1280, n)
    return m
    
def densenet121(n, weight = None):
    m = models.densenet121()
    if weight:
        
        # torchvision.models.densenet._load_state_dict 구현 참고
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(weight.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                weight[new_key] = weight[key]
                del weight[key]
                
        m.load_state_dict(weight)
        freeze(m)
        
    m.classifier = nn.Linear(1024, n)
    return m
    
def efficientnet_v2_s(n, weight = None):
    m = models.efficientnet_v2_s()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[1] = nn.Linear(1280, n)
    return m

def resnet50(n, weight = None):
    m = models.resnet50()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.fc = nn.Linear(2048, n)
    return m
    
def inception_v3(n, weight = None):
    m = models.inception_v3(aux_logits=False, init_weights=False)
    if weight:
        m.load_state_dict(weight, strict=False)
        freeze(m)
    m.fc = nn.Linear(2048, n)
    return m
    
def swin_t(n, weight = None):
    m = models.swin_t()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.head = nn.Linear(768, n)
    return m
    
def vit_b_16(n, weight = None):
    m = models.vit_b_16()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.heads.head = nn.Linear(768, n)
    return m

def convnext_base(n, weight = None):
    m = models.convnext_base()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[2] = nn.Linear(1024, n)
    return m

def efficientnet_b4(n, weight = None):
    m = models.efficientnet_b4()
    if weight:
        m.load_state_dict(weight)
        freeze(m)
    m.classifier[1] = nn.Linear(1792, n)
    return m
    
    
    
    
    
    
    
# 모델의 가중치, 전처리 함수등을 임포트 할수있는 파이토치 내장 모델 객체 저장 딕셔너리     
MODEL_OBJECTS = {
    
    'squeezenet1_1' : models.SqueezeNet1_1_Weights.DEFAULT,
    'shufflenet_v2_x1_0' : models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
    'mobilenet_v3_small' : models.MobileNet_V3_Small_Weights.DEFAULT,
    'mobilenet_v3_large' : models.MobileNet_V3_Large_Weights.DEFAULT,
    'densenet121' : models.DenseNet121_Weights.DEFAULT,
    'efficientnet_v2_s' : models.EfficientNet_V2_S_Weights.DEFAULT,
    'resnet50' : models.ResNet50_Weights.DEFAULT,
    'inception_v3' : models.Inception_V3_Weights.DEFAULT,
    'swin_t' : models.Swin_T_Weights.DEFAULT,
    'vit_b_16' : models.ViT_B_16_Weights.DEFAULT,
    'convnext_base' : models.ConvNeXt_Base_Weights.DEFAULT,
    'efficientnet_b4' : models.EfficientNet_B4_Weights.DEFAULT
    
}
    
# 모델 크기별 분류 
model_cards = {
    'tiny': [
        [squeezenet1_1, MODEL_OBJECTS['squeezenet1_1']],
        [shufflenet_v2_x1_0, MODEL_OBJECTS['shufflenet_v2_x1_0']],
        [mobilenet_v3_small, MODEL_OBJECTS['mobilenet_v3_small']],
    ],
    'small': [
        [mobilenet_v3_large, MODEL_OBJECTS['mobilenet_v3_large']],
        [densenet121, MODEL_OBJECTS['densenet121']],
        [efficientnet_v2_s, MODEL_OBJECTS['efficientnet_v2_s']]
    ],
    'base': [
        [resnet50, MODEL_OBJECTS['resnet50']],
        [inception_v3, MODEL_OBJECTS['inception_v3']],
        [swin_t, MODEL_OBJECTS['swin_t']]
    ],
    'large': [
        [vit_b_16, MODEL_OBJECTS['vit_b_16']],
        [convnext_base, MODEL_OBJECTS['convnext_base']],
        [efficientnet_b4, MODEL_OBJECTS['efficientnet_b4']]
    ]
}

