import torch
import torch.nn as nn
from transformers import ASTModel

class EPICASTBaseline(nn.Module):
    """
    Modello Baseline per EPIC-Sounds basato su Audio Spectrogram Transformer.
    Sfrutta pesi pre-addestrati su AudioSet per l'estrazione delle feature
    e implementa una custom classification head per il task specifico.
    """
    def __init__(self, num_classes: int, pretrained_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.ast = ASTModel.from_pretrained(pretrained_model_name)
        hidden_size = self.ast.config.hidden_size
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definisce il forward pass (flusso dei tensori) attraverso la rete.
        
        Args:
            x (torch.Tensor): Tensore di input proveniente dal DataLoader.
                              Shape attesa: [Batch, Channels, Mel_bins, Frames] -> [B, 1, 128, 1024]
        Returns:
            torch.Tensor: Logits di output. Shape: [Batch, num_classes]
        """

        x = x.squeeze(1) 
        
        x = x.transpose(1, 2).contiguous()
        
        outputs = self.ast(input_values=x)
        
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_token)
        
        return logits