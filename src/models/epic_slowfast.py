import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50


class EpicSlowFast(nn.Module):
    """
    SlowFast-R50 (pytorchvideo, pre-addestrato su Kinetics-400).

    SlowFast usa due percorsi e vuole in input una lista [slow, fast]:
      - fast: tutti i T frame (alta risoluzione temporale, bassa capacita')
      - slow: 1 frame ogni alpha (bassa risoluzione temporale, alta capacita')
    Il dataset fornisce un singolo clip (B, T, C, H, W); il "PackPathway" (split slow/fast)
    e la permute sono fatti qui dentro.

    Feature concatenata slow+fast = 2304-d. Teste: verb 97, noun 300.
    num_frames DEVE essere divisibile per alpha (es. T=32, alpha=4 -> slow=8).
    """

    def __init__(self, num_verbs, num_nouns, weights='slowfast_r50_kinetics.pth',
                 dropout=0.5, alpha=4):
        super().__init__()
        self.alpha = alpha

        backbone = slowfast_r50(pretrained=False)
        if weights:
            state = torch.load(weights, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            backbone.load_state_dict(state)

        D = backbone.blocks[-1].proj.in_features      # 2304
        backbone.blocks[-1].proj = nn.Identity()      # espone la feature

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(D, num_verbs)
        self.noun_head = nn.Linear(D, num_nouns)

    def _pack_pathway(self, x):
        # x: (B, C, T, H, W) -> [slow (T/alpha), fast (T)]
        fast = x
        slow = x[:, :, ::self.alpha].contiguous()
        return [slow, fast]

    def forward(self, x):
        # x dal dataset: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feats = self.backbone(self._pack_pathway(x))   # (B, 2304)
        feats = self.dropout(feats)
        return self.verb_head(feats), self.noun_head(feats)
