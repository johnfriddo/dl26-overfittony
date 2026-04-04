import os
import time
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.ast import EPICASTBaseline

def measure_model_stats(model: torch.nn.Module, device: torch.device, input_shape: tuple = (1, 1, 128, 1024)):
    """
    Calcola la dimensione del modello in MB e il tempo di inferenza medio in ms.
    """

    temp_file = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_file)
    size_mb = os.path.getsize(temp_file) / (1024 * 1024)
    os.remove(temp_file)  # Pulizia immediata
    

    model.eval()
    model.to(device)
    

    dummy_input = torch.randn(input_shape).to(device)
    

    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input)
            
    # --- FASE DI MISURAZIONE ---
    num_iterations = 100
    
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

    avg_time_ms = total_time_ms / num_iterations

    return size_mb, avg_time_ms

if __name__ == "__main__":
    num_classes = 44
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    print("Inizializzazione del modello AST...")
    model = EPICASTBaseline(num_classes=num_classes)
    
    print(f"Misurazione in corso sul dispositivo: {device}...")
    size_mb, avg_time_ms = measure_model_stats(model, device)
    
    print("\n" + "=" * 40)
    print("RISULTATI BASELINE AST")
    print("=" * 40)
    print(f"Model Size (su disco)     : {size_mb:.2f} MB")
    print(f"Inference Time (1 sample) : {avg_time_ms:.2f} ms")
    print("=" * 40 + "\n")