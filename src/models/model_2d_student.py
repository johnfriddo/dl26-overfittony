import torch
import torch.nn as nn
from torchvision import models

class EpicResNet(nn.Module):
    """
    Tre strategie di fusione temporale su backbone ResNet50 2D:
    - 'late_pool', 'late_fc', 'early'
    Le etichette restano gli ID ufficiali EPIC: teste a 97 (verb) e 300 (noun).
    """
    def __init__(self, num_verbs, num_nouns, weights='resnet50_pesi.pth',
                 dropout=0.5, fusion='late_pool', num_frames=8, fc_hidden=512):
        super().__init__()
        self.fusion = fusion
        self.num_frames = num_frames

        backbone = models.resnet50(weights=None)
        if weights:
            state = torch.load(weights, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            backbone.load_state_dict(state)

        D = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()

        if fusion == 'early':
            old = backbone.conv1
            new = nn.Conv2d(3 * num_frames, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=(old.bias is not None))
            with torch.no_grad():
                new.weight.copy_(old.weight.repeat(1, num_frames, 1, 1) / num_frames)
            backbone.conv1 = new

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)

        if fusion == 'late_fc':
            self.fuse = nn.Sequential(
                nn.Linear(D * num_frames, fc_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_in = fc_hidden
        else:
            self.fuse = None
            head_in = D

        self.verb_head = nn.Linear(head_in, num_verbs)
        self.noun_head = nn.Linear(head_in, num_nouns)

    def forward(self, x):
        if self.fusion == 'early':
            b, t, c, h, w = x.shape
            feats = self.backbone(x.reshape(b, t * c, h, w))         
            feats = self.dropout(feats)

        elif self.fusion == 'late_fc':
            b, t, c, h, w = x.shape
            feats = self.backbone(x.reshape(b * t, c, h, w))         
            feats = feats.reshape(b, t * feats.size(-1))             
            feats = self.fuse(feats)                                  

        else:  # late_pool
            if x.dim() == 5:
                b, t, c, h, w = x.shape
                feats = self.backbone(x.reshape(b * t, c, h, w)).reshape(b, t, -1).mean(1)
            else:
                feats = self.backbone(x)
            feats = self.dropout(feats)

        # Ritorna i logit delle teste e il vettore di feature intermedio (2048)
        return self.verb_head(feats), self.noun_head(feats), feats