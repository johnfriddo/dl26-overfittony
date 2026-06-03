import os
import sys
import torch
import torch.nn as nn


base_path = os.path.dirname(__file__)
efficientat_dir = os.path.abspath(os.path.join(base_path, 'EfficientAT'))


original_cwd = os.getcwd()

os.chdir(efficientat_dir)
sys.path.append(efficientat_dir)


from models.mn.model import get_model

os.chdir(original_cwd)

class EPICEfficientATBaseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        # pretrained_name=None gli impedisce di cercare internet.
        self.backbone = get_model(width_mult=1.0, pretrained_name=None) 
        
        local_weights_path = os.path.join(base_path, 'EfficientAT', 'resources', 'mn10_as_mAP_471.pt')
        checkpoint = torch.load(local_weights_path, map_location='cpu')
        
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.backbone.load_state_dict(state_dict, strict=False)

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, tuple):
            return out[0]
        return out