import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import PreconvolutedANCDataset, apply_dynamic_path
from model import TimeDomainANC

def evaluate_and_plot(model, test_loader, device, sr=48000, scenario_title="Test", save_prefix="test"):
    """
    Evaluates the Active Noise Control (ANC) model on the test set and calculates 
    the average Noise Reduction (NR) in decibels (dB). Generates both time-domain 
    and frequency-domain visualizations.
    """
    model.eval()
    nr_list = []
    all_d_np = []
    all_e_np = []
    
    with torch.no_grad():
        for i, (x_t, sh, d_t) in enumerate(test_loader):
            x_t, sh, d_t = x_t.to(device), sh.to(device), d_t.to(device)
            
            # --- Model Inference ---
            y_t = model(x_t)
            
            # Pass the predicted anti-noise through the dynamic secondary physical path
            a_t = apply_dynamic_path(y_t, sh)
            
            # Calculate residual error 
            e_t = d_t - a_t 
            
            d_np = d_t[0].cpu().numpy()
            e_np = e_t[0].cpu().numpy()
            all_d_np.append(d_np)
            all_e_np.append(e_np)
            
            # Calculate Noise Reduction (NR) in dB
            energy_d = np.sum(d_np ** 2)
            energy_e = np.sum(e_np ** 2)
            nr_db = 10 * np.log10(energy_d / (energy_e + 1e-12))
            nr_list.append(nr_db)

    avg_nr = np.mean(nr_list)
    print(f"\n>> [{scenario_title}] Evaluation Complete! Independent Scene NR: {[f'{x:.2f}dB' for x in nr_list]}")
    print(f">> [{scenario_title}] Average Scene NR: {avg_nr:.2f} dB")

    num_test_scenarios = len(all_d_np)
    cols = 2
    rows = max(1, (num_test_scenarios + 1) // 2)

    # ==================== Plot 1: Time Domain Waveform Comparison ====================
    fig_time, axes_time = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    if isinstance(axes_time, np.ndarray):
        axes_time = axes_time.flatten()
    else:
        axes_time = [axes_time]
    
    plot_duration = 0.9 
    plot_samples = int(plot_duration * sr)
    time_axis = np.arange(plot_samples) * 1000 / sr 
    
    for i in range(num_test_scenarios):
        axes_time[i].plot(time_axis, all_d_np[i][:plot_samples], label='Primary Noise $d(t)$', color='blue', alpha=0.6)
        axes_time[i].plot(time_axis, all_e_np[i][:plot_samples], label='Residual Noise $e(t)$', color='red', alpha=0.8)
        axes_time[i].set_title(f"Scenario {i+1} Time Domain (First 50ms) | NR: {nr_list[i]:.2f} dB")
        axes_time[i].set_xlabel("Time (ms)")
        axes_time[i].set_ylabel("Amplitude")
        axes_time[i].legend(loc='upper right')
        axes_time[i].grid(True)
        
    for j in range(num_test_scenarios, len(axes_time)):
        axes_time[j].axis('off')
        
    plt.suptitle(f"[{scenario_title}] Time Domain Noise Cancellation | Average NR: {avg_nr:.2f} dB", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    time_save_path = f"anc_{save_prefix}_time_result.png"
    plt.savefig(time_save_path, dpi=300)
    print(f">> Time domain results saved to {time_save_path}")

    # ==================== Plot 2: Frequency Domain PSD Comparison ====================
    fig_freq, axes_freq = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    if isinstance(axes_freq, np.ndarray):
        axes_freq = axes_freq.flatten()
    else:
        axes_freq = [axes_freq]
    
    for i in range(num_test_scenarios):
        axes_freq[i].psd(all_d_np[i], NFFT=1024, Fs=sr, label='Primary Noise', color='blue', alpha=0.6)
        axes_freq[i].psd(all_e_np[i], NFFT=1024, Fs=sr, label='Residual Noise', color='red', alpha=0.8)
        axes_freq[i].set_title(f"Scenario {i+1} Frequency Spectrum")
        axes_freq[i].set_xlabel("Frequency (Hz)")
        axes_freq[i].set_ylabel("Power/Frequency (dB/Hz)")
        axes_freq[i].legend()
        axes_freq[i].grid(True, which="both", ls="-", alpha=0.5)
        axes_freq[i].set_xscale('log')
        axes_freq[i].set_xlim(left=20, right=sr/2)
        
    for j in range(num_test_scenarios, len(axes_freq)):
        axes_freq[j].axis('off')
        
    plt.suptitle(f"[{scenario_title}] Frequency Domain PSD (0 to {sr//2}Hz)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    freq_save_path = f"anc_{save_prefix}_freq_result.png"
    plt.savefig(freq_save_path, dpi=300)
    print(f">> Frequency domain results saved to {freq_save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = "dataset"
    noise_dir = os.path.join(dataset_dir, "NOISE")
    
    print("Scanning noise directory architecture...")
    all_noise_names = sorted([os.path.splitext(f)[0] for f in os.listdir(noise_dir) if f.endswith('.wav')])
    
    if not all_noise_names:
        print("Error: No valid noise files found. Terminating program.")
        return

    # Separate scenes: Reserve the final two scenes for the testing phase
    if len(all_noise_names) >= 3:
        train_noises = all_noise_names[:-2]
        test_noises = all_noise_names[-2:] 
        print(f"Scene allocation successful. Training noise sources: {len(train_noises)}, Test spliced sources: {len(test_noises)}")
    else:
        train_noises = all_noise_names
        test_noises = all_noise_names
        print("Warning: Insufficient noise files (less than 3). Testing phase will degrade and reuse training data.")

    # Static acoustic path indices allocation
    num_paths = 10 
    train_count = int(num_paths * 0.8)
    
    train_path_indices = list(range(0, train_count))
    test_path_indices = list(range(train_count, num_paths))
    print(f"Path allocation successful. Training paths: {len(train_path_indices)}, Test paths: {len(test_path_indices)}")

    # Instantiate Datasets and Dataloaders
    train_dataset = PreconvolutedANCDataset(dataset_dir, train_noises, train_path_indices, segment_duration=1.0, sr=48000, is_train=True)
    test_dataset_seen_paths = PreconvolutedANCDataset(dataset_dir, test_noises, train_path_indices, segment_duration=1.0, sr=48000, is_train=False)
    test_dataset_unseen_paths = PreconvolutedANCDataset(dataset_dir, test_noises, test_path_indices, segment_duration=1.0, sr=48000, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader_seen = DataLoader(test_dataset_seen_paths, batch_size=1, shuffle=False)
    test_loader_unseen = DataLoader(test_dataset_unseen_paths, batch_size=1, shuffle=False)

    # Initialize the Time-Domain ANC model
    model = TimeDomainANC(in_channels=1, out_channels=1, hidden_channels=32, num_layers=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    epochs = 20

    print("\n=== Initiating Multi-Scene Acoustic Path Training ===")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_t, sh, d_t in train_loader:
            x_t, sh, d_t = x_t.to(device), sh.to(device), d_t.to(device)
            optimizer.zero_grad()
            
            # --- Model Inference ---
            # The network processes raw audio waveforms directly
            y_t = model(x_t)
            
            # --- Anti-noise passed through the dynamic secondary physical path ---
            a_t = apply_dynamic_path(y_t, sh)
            
            # --- Error Calculation ---
            e_t = d_t - a_t 
            loss = torch.mean(e_t ** 2) 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {train_loss/len(train_loader):.6f}")

    # 5. Testing and Evaluation Subroutines
    print("\n=== Initiating Test 1: Unseen Noise + Seen Paths (Training Paths) ===")
    evaluate_and_plot(model, test_loader_seen, device, sr=48000, 
                      scenario_title="Unseen Noise, Seen Paths", 
                      save_prefix="seen_paths")

    print("\n=== Initiating Test 2: Unseen Noise + Unseen Paths (Validation Paths) ===")
    evaluate_and_plot(model, test_loader_unseen, device, sr=48000, 
                      scenario_title="Unseen Noise, Unseen Paths", 
                      save_prefix="unseen_paths")

    plt.show()

if __name__ == "__main__":
    main()