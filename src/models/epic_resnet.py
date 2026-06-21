import torch
import torch.nn as nn
from torchvision import models


class EpicResNet(nn.Module):
    """
    - 'late_pool': 2D CNN su ogni frame -> media delle feature sui T frame -> testa.
                   E' la Late Fusion con pooling (consenso stile TSN). Default.
    - 'late_fc'  : 2D CNN su ogni frame -> concatena le T feature -> MLP -> testa.
                   (Karpathy 2014). T e' FISSO dall'architettura.
    - 'early'    : impila i T frame sui canali (3T) -> un'unica 2D CNN col primo conv
                   modificato a 3T canali. (Karpathy 2014). T e' FISSO.

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
        # x: (B, T, C, H, W); per late_pool e' ammesso anche (B, C, H, W) single-frame
        if self.fusion == 'early':
            b, t, c, h, w = x.shape
            feats = self.backbone(x.reshape(b, t * c, h, w))         # (B, D)
            feats = self.dropout(feats)

        elif self.fusion == 'late_fc':
            b, t, c, h, w = x.shape
            feats = self.backbone(x.reshape(b * t, c, h, w))         # (B*T, D)
            feats = feats.reshape(b, t * feats.size(-1))             # (B, T*D)
            feats = self.fuse(feats)                                 # (B, fc_hidden), dropout incluso

        else:  # late_pool (= TSN, media delle feature nel tempo)
            if x.dim() == 5:
                b, t, c, h, w = x.shape
                feats = self.backbone(x.reshape(b * t, c, h, w)).reshape(b, t, -1).mean(1)
            else:
                feats = self.backbone(x)
            feats = self.dropout(feats)

        return self.verb_head(feats), self.noun_head(feats)
