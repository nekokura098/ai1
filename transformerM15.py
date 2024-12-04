import MetaTrader5 as mt5
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy
class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads, hidden_dim, n_layers, output_dim, sequence_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_dim = input_dim

        # Positional Encoding (Register sebagai buffer)
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
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def _get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Tambahkan dimensi batch
    def forward(self, x, target):
        # Cek bentuk input
        
        # Pastikan target dalam bentuk [batch, sequence_length]
        batch_size = x.size(0)  # Dimensi batch
        target_length = x.size(1)  # Dimensi sequence_length

        # Ubah bentuk target menjadi [batch_size, sequence_length]
        target = target.unsqueeze(1).repeat(1, target_length)  # Menambah dimensi dan memperbanyak sepanjang sequence length

        # Add positional encoding to input
        pos_encoding = self.positional_encoding[:, :x.size(1), :].to(x.device)
        x = x + pos_encoding

        # Normalize input
        x = self.layer_norm(x)
        
        # Transpose for transformer encoder
        x = x.transpose(0, 1)  # [batch_size, sequence_length, input_dim]
        memory = self.encoder(x)

        # Prepare target
        target_repeated = target.unsqueeze(-1).repeat(1, 1, x.size(-1))  # [batch_size, sequence, input_dim]
        target_pos_encoding = self.positional_encoding[:, :target_repeated.size(1), :].to(target.device)
        target_repeated = target_repeated + target_pos_encoding
        
        # Masking
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_repeated.size(1)).to(target.device)
        target_repeated = target_repeated.transpose(0, 1)  # [sequence_length, batch_size, input_dim]
        output = self.decoder(target_repeated, memory, tgt_mask=tgt_mask)

        # Transpose back
        output = output.transpose(0, 1)  # [batch_size, sequence_length, input_dim]
        output = self.fc(output)

        return output



    def _generate_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
def create_target(y):
    """
    Membuat target untuk data dengan multiple dimensi.
    
    Args:
        y (torch.Tensor): Tensor target asli
    
    Returns:
        torch.Tensor: Target yang digeser
    """
    # Jika tensor 1D, lakukan shift sederhana
    if y.dim() == 1:
        target = torch.zeros_like(y)
        target[:-1] = y[1:]
        return target
    
    # Jika tensor 2D atau lebih, geser pada dimensi terakhir
    target = torch.zeros_like(y)
    
    # Geser semua dimensi kecuali dimensi batch (dimensi pertama)
    target[:, :-1, ...] = y[:, 1:, ...]
    
    return target

def train(model, X_train, y_train, epochs=1, batch_size=32, learning_rate=0.001):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            # Pastikan batch penuh
            if i + batch_size > len(X_train):
                break
            
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Generate shifted target
            target = create_target(y_batch)

            optimizer.zero_grad()

            # Forward pass dengan X_batch dan target
            outputs = model(X_batch, target=target)

            # Squeeze output untuk kehilangan dimensi yang tidak perlu
            outputs = outputs.view(-1, outputs.size(-1))  # Meratakan output menjadi [batch_size * sequence_length, output_dim]
            y_batch = y_batch.view(-1)  # Meratakan target menjadi [batch_size * sequence_length]

            # Menghitung loss
            loss = criterion(outputs, y_batch)
            
            # Backward pass dengan gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
        
        # Step scheduler
        scheduler.step(avg_loss)



def evaluate_model(model, X_test, y_test, target_test, scaler):
    model.eval()
    with torch.no_grad():
        # Forward pass dengan input dan target
        predictions = model(X_test, target=target_test)  # [batch_size, seq_len, num_features]
        
        # Konversi tensor ke numpy dengan float32
        predictions_np = predictions.detach().numpy().astype(np.float32)  # [batch_size, seq_len, num_features]
        y_test_np = y_test.detach().numpy().astype(np.float32)  # [batch_size]
        
        # Rata-rata prediksi sepanjang sequence untuk setiap data
        predictions_avg = np.mean(predictions_np, axis=1)  # [batch_size, num_features]
        
        # Persiapkan array untuk menyimpan rescaled predictions
        predictions_rescaled = []
        
        # Pastikan scaler hanya menerima numeric data
        # Ambil hanya kolom numeric untuk inverse transform
        scale_columns = scaler.feature_names_in_
        
        # Rescaling prediksi
        pred_feature_expanded = predictions_avg[:, :len(scale_columns)]
        # Pastikan array 2D untuk inverse_transform
        if pred_feature_expanded.ndim == 1:
            pred_feature_expanded = pred_feature_expanded.reshape(-1, 1)
        
        # Inverse transform
        pred_rescaled = scaler.inverse_transform(pred_feature_expanded)
        
        # Hitung MSE dan MAE antara prediksi dan target
        mse_loss = np.mean((pred_rescaled[:, 0] - y_test_np)**2)
        mae_loss = np.mean(np.abs(pred_rescaled[:, 0] - y_test_np))
        
        print(f"Mean Squared Error: {mse_loss}")
        print(f"Mean Absolute Error: {mae_loss}")
        
        return pred_rescaled, mse_loss, mae_loss
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model disimpan di {file_path}")

def encode_categorical_features(data):
    # Encode symbol and timeframe
    symbol_mapping = {symbol: idx for idx, symbol in enumerate(data['symbol'].unique())}
    timeframe_mapping = {timeframe: idx for idx, timeframe in enumerate(data['timeframe'].unique())}
    
    # Create new columns with encoded values
    data['symbol_encoded'] = data['symbol'].map(symbol_mapping)
    data['timeframe_encoded'] = data['timeframe'].map(timeframe_mapping)
    
    return data, symbol_mapping, timeframe_mapping

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def main():
    sequence_length = 60
    folder_path = r"C:\Users\USER\Documents\latihan phyton\type 1\project1\contoh\my_chatgpt\Project_01\data"

    try:
        # Scan all CSV files in the folder
        csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
        if not csv_files:
            print("Error: No CSV files found in the folder.")
            return

        all_data = []  # Store data from all files
        
        for file in csv_files:
            print(f"Reading file: {file}")
            data = pd.read_csv(file)
            if data.empty:
                print(f"Warning: File {file} is empty. Skipping...")
                continue
            all_data.append(data)

        # Combine all data into a single DataFrame
        if not all_data:
            print("Error: No data could be loaded from the folder.")
            return
        combined_data = pd.concat(all_data, ignore_index=True)

        # Define required columns
        required_columns = [
            'open', 'high', 'low', 'close', 'MA_30', 'MA_60', 'RSI', 
            'MACD', 'Signal_Line', 'Upper_Band', 'Middle_Band', 'Lower_Band', 
            'Stochastic_K', 'Stochastic_D', 'symbol', 'timeframe'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in combined_data.columns]
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            print(f"Available columns: {list(combined_data.columns)}")
            return

        # Prepare features for scaling
        features_for_scaling = combined_data[required_columns[:-2]]  # Exclude 'symbol' and 'timeframe'

        # Initialize and fit the scaler BEFORE creating sequences
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features_for_scaling)

        # Add back 'symbol' and 'timeframe' without scaling
        scaled_data = pd.DataFrame(features_scaled, columns=features_for_scaling.columns)
        scaled_data['symbol'] = combined_data['symbol']
        scaled_data['timeframe'] = combined_data['timeframe']

        # Convert to numpy array
        features = scaled_data.values
        target = combined_data['close'].values
        print("Features shape:", features.shape)
        print("Features dtype:", features.dtype)
        print("Target shape:", target.shape)
        print("Target dtype:", target.dtype)

        # Prepare sequences
        def create_sequences(data, target, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(target[i + seq_length])
            return np.array(X), np.array(y)

        X_data, y_data = create_sequences(features, target, sequence_length)

        # Split data
        split_ratio = 0.8
        split_index = int(len(X_data) * split_ratio)
        
        X_train, X_test = X_data[:split_index], X_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Initialize model with correct input dimension
        model = TransformerModel(
            input_dim=features.shape[1],  # Dynamic input dimension
            n_heads=8, 
            hidden_dim=32, 
            n_layers=4, 
            output_dim=1,
            sequence_length=sequence_length
        )

        # Train model
        train(model, X_train, y_train)

        # Evaluate model (with scaler passed correctly)
        predictions, mse, mae = evaluate_model(model, X_test, y_test, y_test, scaler)

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

if __name__ == "__main__":
    main()


