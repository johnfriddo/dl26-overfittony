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
        
        # 1. Caricamento del modello base (solo feature extractor, senza layer di classificazione)
        self.ast = ASTModel.from_pretrained(pretrained_model_name)
        
        # 2. Estrazione della dimensionalità dell'embedding latente (default: 768)
        hidden_size = self.ast.config.hidden_size
        
        # 3. Definizione della Classification Head
        # Proietta il vettore latente (768) nello spazio delle classi di EPIC-Sounds
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definisce il forward pass (flusso dei tensori) attraverso la rete.
        
        Args:
            x (torch.Tensor): Tensore di input proveniente dal DataLoader.
                              Shape attesa: [Batch, Channels, Mel_bins, Frames] -> [B, 1, 128, 512]
        Returns:
            torch.Tensor: Logits di output. Shape: [Batch, num_classes]
        """
        # A. ADATTAMENTO DIMENSIONALE
        # L'AST di Hugging Face si aspetta una shape [Batch, Frames, Mel_bins]
        
        # Rimuoviamo la dimensione del canale mono (dim=1) -> [B, 128, 512]
        x = x.squeeze(1) 
        
        # Trasponiamo l'asse delle frequenze con l'asse temporale -> [B, 512, 128]
        x = x.transpose(1, 2).contiguous()
        
        # B. ESTRAZIONE DELLE FEATURE
        # Il modello divide lo spettrogramma in patch, somma i Positional Embeddings 
        # ed elabora la sequenza tramite i blocchi di Self-Attention.
        outputs = self.ast(input_values=x)
        
        # C. AGGREGAZIONE GLOBALE
        # L'output 'last_hidden_state' ha shape [Batch, Sequence_Length, Hidden_Size]
        # Estraiamo l'embedding associato al [CLS] token, che si trova sempre in posizione 0 
        # sull'asse della sequenza. Questo vettore condensa l'informazione globale dell'audio.
        cls_token = outputs.last_hidden_state[:, 0, :] # Shape: [B, 768]
        
        # D. CLASSIFICAZIONE
        # Generiamo le probabilità non normalizzate (logits)
        logits = self.classifier(cls_token)
        
        return logits