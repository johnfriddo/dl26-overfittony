import os
import time
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.ast import EPICASTBaseline
from src.models.efficientat import EPICEfficientATBaseline

def measure_model_stats(model: torch.nn.Module, device: torch.device, input_shape: tuple = (1, 1, 128, 1024)):
    """Calcola la dimensione in MB e l'inference time in ms."""
    temp_file = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_file)
    size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    os.remove(temp_file)

    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup della GPU
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input)
            
    num_iterations = 100
    
    # Fase di misurazione
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize() 
        total_time_ms = start_event.elapsed_time(end_event)
    else: 
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

    return size_mb, total_time_ms / num_iterations

if __name__ == "__main__":
    num_classes = 44
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione benchmark sul dispositivo: {device}\n")
    
    # Dizionario con tutti i modelli da testare
    models_to_test = {
        "AST (Baseline Pesante)": EPICASTBaseline(num_classes=num_classes),
        "EfficientAT (Baseline Edge)": EPICEfficientATBaseline(num_classes=num_classes)
    }

    print("=" * 65)
    print(f"{'Modello':<30} | {'Size (MB)':<12} | {'Inference (ms)':<15}")
    print("=" * 65)


    for name, model in models_to_test.items():
        size_mb, avg_time_ms = measure_model_stats(model, device)
        print(f"{name:<30} | {size_mb:>10.2f} MB | {avg_time_ms:>12.2f} ms")
        
    print("=" * 65 + "\n")