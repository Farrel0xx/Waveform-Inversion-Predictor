import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import psutil
import logging
import gc

# Setup logging biar bisa monitor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fungsi buat monitor RAM
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

log_memory_usage()

# 1. Load Data Training (Cuma Ambil 1 Sample, Pake memmap biar hemat RAM)
data_path = '/home/ghost00/Videos/Project-Kaggle/train_samples/Vel/data/data1.npy'
model_path = '/home/ghost00/Videos/Project-Kaggle/train_samples/Vel/model/model1.npy'

logging.info("Loading seismic data (1 sample only, using memmap)...")
seismic_data_full = np.load(data_path, mmap_mode='r')
seismic_data = seismic_data_full[:1].copy()  # Ambil cuma 1 sample
del seismic_data_full
gc.collect()
log_memory_usage()

logging.info("Loading velocity maps (1 sample only, using memmap)...")
velocity_maps_full = np.load(model_path, mmap_mode='r')
velocity_maps = velocity_maps_full[:1].copy()  # Ambil cuma 1 sample
del velocity_maps_full
gc.collect()
log_memory_usage()

# 2. Siapin data buat training
X = seismic_data  # Shape: (1, num_sources, time_steps, num_receivers)
y = velocity_maps  # Shape: (1, height, width)
X = X[..., np.newaxis]  # Shape: (1, num_sources, time_steps, num_receivers, 1)

del seismic_data
gc.collect()
log_memory_usage()

# 3. Baseline Model: CNN yang Super Duper Sederhana
def simple_cnn(input_shape, output_shape):
    model = models.Sequential([
        layers.Conv3D(4, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),  # Filter cuma 4
        layers.Flatten(),
        layers.Dense(16, activation='relu'),  # Dense layer cuma 16
        layers.Dense(output_shape[0] * output_shape[1], activation='linear'),
        layers.Reshape(output_shape)
    ])
    # Pake MeanSquaredError sebagai loss function
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model

input_shape = (X.shape[1], X.shape[2], X.shape[3], 1)
output_shape = (y.shape[1], y.shape[2])

logging.info("Building model...")
model = simple_cnn(input_shape, output_shape)
log_memory_usage()

# 4. Training (Batch Size 1, Epochs 1)
logging.info("Starting training...")
model.fit(X, y, epochs=1, batch_size=1, validation_split=0.0)
log_memory_usage()

# 5. Simpan Model
logging.info("Saving model...")
model.save('/home/ghost00/Videos/Project-Kaggle/waveform_inversion_model.h5')
print("Model saved to waveform_inversion_model.h5")
log_memory_usage()

# Bersihin memory
del X, y, model
gc.collect()
log_memory_usage()
