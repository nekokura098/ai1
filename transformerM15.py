import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from tqdm import tqdm
import math
from torch.cuda.amp import GradScaler, autocast


# Save model checkpoint
def save_model(model, optimizer, epoch, path="model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


# Load model checkpoint
def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, last trained epoch: {epoch}")
    return model, optimizer, epoch


# Transformer model class
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, hidden_dim, n_layers, output_dim, sequence_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim

        # Positional Encoding
        pe = self._get_positional_encoding(sequence_length, input_dim)
        self.register_buffer('positional_encoding', pe)

        # Normalization layer
        self.layer_norm = nn.LayerNorm(input_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Fully connected layer for prediction
        self.fc = nn.Linear(input_dim * sequence_length, output_dim)

    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = x + pos_encoding

        # Normalize input
        x = self.layer_norm(x)

        # Transpose for encoder
        x = x.transpose(0, 1)  # [batch_size, sequence_length, input_dim] => [sequence_length, batch_size, input_dim]

        # Encoder processing
        memory = self.encoder(x)

        # Transpose back and flatten
        memory = memory.transpose(0, 1)  # [sequence_length, batch_size, input_dim] => [batch_size, sequence_length, input_dim]
        memory = memory.reshape(memory.size(0), -1)

        # Final prediction
        output = self.fc(memory)

        return output


# Train function
def train(model, X_train, y_train, epochs=5, batch_size=64, learning_rate=0.001, save_path="model_checkpoint.pth"):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scaler = GradScaler()  # For Mixed Precision
    best_loss = float('inf')  # Track best loss for saving the model

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_end = min(i + batch_size, len(X_train))
            X_batch = X_train[i:batch_end]
            y_batch = y_train[i:batch_end]

            optimizer.zero_grad()

            with autocast():  # Mixed Precision
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

        # Save model if it achieves a new best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, path=save_path)

def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        predictions = model(X_test).cpu().numpy()

        # Pastikan prediksi berbentuk 1D atau 2D single column
        predictions = predictions.reshape(-1, 1)

        # Persiapkan array dummy dengan bentuk yang sama dengan data training
        dummy_predictions = np.zeros((predictions.shape[0], X_test.shape[-1]))
        dummy_predictions[:, 0] = predictions.squeeze()  # Simpan prediksi di kolom pertama

        # Inverse transform dengan dummy array
        pred_rescaled = scaler.inverse_transform(dummy_predictions)[:, 0]

        # Konversi y_test ke numpy 
        y_test_np = y_test.cpu().numpy()

        # Hitung metrik evaluasi
        mse_loss = np.mean((pred_rescaled - y_test_np)**2)
        mae_loss = np.mean(np.abs(pred_rescaled - y_test_np))

        # Tambahkan metrik tambahan
        mape_loss = np.mean(np.abs((y_test_np - pred_rescaled) / y_test_np)) * 100
        rmse_loss = np.sqrt(mse_loss)

        # Cetak semua metrik
        print(f"Mean Squared Error (MSE): {mse_loss}")
        print(f"Root Mean Squared Error (RMSE): {rmse_loss}")
        print(f"Mean Absolute Error (MAE): {mae_loss}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_loss}%")

        return pred_rescaled, mse_loss, mae_loss

def main():
    sequence_length = 120
    folder_path = r"data"
    final_model_path = "final_model.pth"  # Path to save the final model
    
    # Determine device (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
        if not csv_files:
            print("Error: No CSV files found in the folder.")
            return
        
        required_columns = [
            'open', 'high', 'low', 'close', 'MA_30', 'MA_60', 'RSI',
            'MACD', 'Signal_Line', 'Upper_Band', 'Middle_Band', 'Lower_Band',
            'Stochastic_K', 'Stochastic_D', 'symbol', 'timeframe'
        ]
        
        # Initialize model
        model = TransformerModel(
            input_dim=len(required_columns),
            n_heads=8,
            hidden_dim=64,
            n_layers=4,
            output_dim=1,
            sequence_length=sequence_length
        ).to(device)
        
        checkpoint_path = "model_checkpoint.pth"
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        start_epoch = 0
        
        # Load model if checkpoint exists
        if os.path.exists(checkpoint_path):
            print("Checkpoint found, loading model...")
            model, optimizer, start_epoch = load_model(checkpoint_path, model, optimizer)
        
        # Lists to store overall performance metrics
        all_mse = []
        all_mae = []
        
        # Process each file individually
        for file in tqdm(csv_files, desc="Processing files", unit="file"):
            print(f"Processing file: {file}")
            data = pd.read_csv(file)
            
            if data.empty:
                print(f"Skipping empty file: {file}")
                continue
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Skipping file {file} due to missing columns: {missing_columns}")
                continue
            
            # Scale features
            features = data[required_columns]
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
            target = data['close'].values
            
            # Create sequences
            def create_sequences(data, target, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:i + seq_length])
                    y.append(target[i + seq_length])
                return np.array(X), np.array(y)
            
            X_data, y_data = create_sequences(features_scaled, target, sequence_length)
            
            # Split data
            split_ratio = 0.8
            split_index = int(len(X_data) * split_ratio)
            X_train, X_test = X_data[:split_index], X_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]
            
            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float16).to(device)
            X_test = torch.tensor(X_test, dtype=torch.float16).to(device)
            y_train = torch.tensor(y_train, dtype=torch.float16).to(device)
            y_test = torch.tensor(y_test, dtype=torch.float16).to(device)
            
            # Train model on the current file
            train(model, X_train, y_train, epochs=50, save_path=checkpoint_path)
            
            # Evaluate model
            predictions, mse, mae = evaluate_model(model, X_test, y_test, scaler)
            print(f"File: {file} - MSE: {mse}, MAE: {mae}")
            
            # Store performance metrics
            all_mse.append(mse)
            all_mae.append(mae)
        
        # Calculate and print overall performance
        print("\nOverall Performance:")
        print(f"Average MSE: {np.mean(all_mse)}")
        print(f"Average MAE: {np.mean(all_mae)}")
        
        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_mse': np.mean(all_mse),
            'avg_mae': np.mean(all_mae)
        }, final_model_path)
        
        print(f"\nFinal model saved to {final_model_path}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()