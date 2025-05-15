# ANN Optimization Script for Pretreatment Process (3 Inputs: NaOH, Time, SSR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from geneticalgorithm import geneticalgorithm as ga
import os
import gc
import shutil
import tempfile
import sys

# Print Python and package versions for debugging
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# Create temporary directory for TensorFlow files
temp_dir = tempfile.mkdtemp()
os.environ['TMPDIR'] = temp_dir
os.makedirs(temp_dir, exist_ok=True)  # Ensure temp directory exists

# Enable memory cleanup
tf.keras.backend.clear_session()
gc.collect()

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
print("\nLoading and preprocessing data...")

# Load data from Excel file
excel_file = 'experiment datafor model.xlsx'
if not os.path.exists(excel_file):
    print(f"Error: Excel file '{excel_file}' not found in directory: {os.getcwd()}")
    raise FileNotFoundError(f"Excel file '{excel_file}' not found.")

try:
    # Load the Excel file
    df_exp = pd.read_excel(excel_file)
    print(f"\nOriginal columns found:", df_exp.columns.tolist())
    
    # Clean column names: remove whitespace, lowercase
    df_exp.columns = df_exp.columns.str.strip().str.lower()
    
    # Drop unnamed columns
    df_exp = df_exp.loc[:, ~df_exp.columns.str.contains('^unnamed')]
    
    # Map common column variations to standard names
    column_mapping = {
        'biomass code': 'biomass_code:',
        'biomass_code': 'biomass_code:',
        'ssr (solid:solvent)': 'ssr (solid:solvent)',
        'ssr': 'ssr (solid:solvent)',
        'solid:solvent ratio': 'ssr (solid:solvent)',
        'naoh (%)': 'naoh (%)',
        'naoh': 'naoh (%)',
        'time (min)': 'time (min)',
        'time': 'time (min)',
        'cellulose retention (%)': 'retention (%)',
        'retention (%)': 'retention (%)',
        'delignification (%)': 'delignification (%)',
        'delignification': 'delignification (%)'
    }
    
    # Rename columns if they exist
    df_exp.rename(columns=column_mapping, inplace=True)
    
    print("\nColumns after renaming:", df_exp.columns.tolist())
    
    # Check for required columns
    required_columns = ['biomass_code:', 'ssr (solid:solvent)', 'naoh (%)', 
                       'time (min)', 'retention (%)', 'delignification (%)']
    
    missing_columns = [col for col in required_columns if col not in df_exp.columns]
    if missing_columns:
        print(f"\nError: Missing required columns: {missing_columns}")
        print("Available columns:", df_exp.columns.tolist())
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Verify data types and handle any conversion issues
    numeric_columns = ['ssr (solid:solvent)', 'naoh (%)', 'time (min)', 
                      'retention (%)', 'delignification (%)']
    
    for col in numeric_columns:
        df_exp[col] = pd.to_numeric(df_exp[col], errors='coerce')
        null_count = df_exp[col].isnull().sum()
        if null_count > 0:
            print(f"\nWarning: {null_count} null values found in {col}")
    
    # Remove any rows with NaN values
    original_len = len(df_exp)
    df_exp = df_exp.dropna(subset=required_columns)
    if len(df_exp) < original_len:
        print(f"\nRemoved {original_len - len(df_exp)} rows with missing values")
    
    print(f"\nFinal data shape: {df_exp.shape}")

except Exception as e:
    print(f"\nError during data loading: {str(e)}")
    raise

# Filter data for rice husk with case-insensitive matching
print("\nFiltering data for rice husk...")
biomass_codes = df_exp['biomass_code:'].str.lower().unique()
print("Available biomass codes:", biomass_codes)

df_rh = df_exp[df_exp['biomass_code:'].str.lower().str.contains('rh|rice', case=False, na=False)].copy()
if len(df_rh) == 0:
    print("Error: No data found for rice husk")
    print("Available biomass codes:", biomass_codes)
    raise ValueError("No data found for rice husk in the dataset")

print(f"Found {len(df_rh)} samples for rice husk")

#####////// Normalize features and targets with error checking/////////
try:
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler()
    
    # Prepare feature matrix
    features = ['naoh (%)', 'time (min)', 'ssr (solid:solvent)']
    X = df_rh[features].values
    if np.isnan(X).any():
        raise ValueError("NaN values found in feature matrix")
    
    # Prepare target matrix
    targets = ['retention (%)', 'delignification (%)']
    y = df_rh[targets].values
    if np.isnan(y).any():
        raise ValueError("NaN values found in target matrix")
    
    # Scale the data
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    def generate_synthetic_data(X, y, n_synthetic=100):
        """Generate synthetic data points using interpolation and small random variations"""
        synthetic_X = []
        synthetic_y = []
        n_samples = X.shape[0]
        
        for _ in range(n_synthetic):
            # Randomly select two real data points
            idx1, idx2 = np.random.randint(0, n_samples, 2)
            t = np.random.random()
            
            # Interpolate between points
            new_X = X[idx1] * t + X[idx2] * (1-t)
            new_y = y[idx1] * t + y[idx2] * (1-t)
            
            # Add small random variations (±5%)
            new_X += np.random.normal(0, 0.05, size=X.shape[1])
            new_y += np.random.normal(0, 0.05, size=y.shape[1])
            
            # Ensure bounds are respected
            new_X = np.clip(new_X, -1, 1)
            new_y = np.clip(new_y, 0, 1)
            
            synthetic_X.append(new_X)
            synthetic_y.append(new_y)
        
        return np.array(synthetic_X), np.array(synthetic_y)

    # Generate synthetic data
    print("\nGenerating synthetic data points...")
    X_synthetic, y_synthetic = generate_synthetic_data(X_scaled, y_scaled)

    # Combine original and synthetic data
    X_combined = np.vstack([X_scaled, X_synthetic])
    y_combined = np.vstack([y_scaled, y_synthetic])

    print(f"Original dataset size: {X_scaled.shape[0]} samples")
    print(f"After augmentation: {X_combined.shape[0]} samples")

    # Update train-test split to use augmented data
    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Print feature ranges
    print("\nFeature ranges after scaling:")
    for i, feature in enumerate(features):
        print(f"{feature}: [{X_scaled[:, i].min():.2f}, {X_scaled[:, i].max():.2f}]")
    
except Exception as e:
    print(f"Error during data scaling: {str(e)}")
    raise

# Force garbage collection
gc.collect()

# ---------------------------
# 2. ANN Model Architecture
# ---------------------------
def build_ann():
    model = tf.keras.Sequential([
        # Input layer with leaky ReLU
        tf.keras.layers.Dense(12, input_shape=(3,)),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        # Hidden layer 1
        tf.keras.layers.Dense(8),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        # Hidden layer 2
        tf.keras.layers.Dense(6),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer (linear activation for regression)
        tf.keras.layers.Dense(2)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_ann()
print(model.summary())

# ---------------------------
# 3. Model Training
# ---------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# Checkpoint callback with cleanup
checkpoint_path = os.path.join(temp_dir, "model_checkpoint.keras")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_best_only=True,
    monitor='val_loss'
)

# Model training with error handling
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=4,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
except Exception as e:
    print(f"Error during model training: {str(e)}")
    raise

def predict_safely(model, X):
    try:
        return model.predict(X, verbose=0)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

def evaluate_model(X, y, name="Set"):
    try:
        y_pred = predict_safely(model, X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"{name} Performance: R2 = {r2:.3f}, MSE = {mse:.5f}")
        return r2, mse
    except Exception as e:
        print(f"Error during {name} evaluation: {str(e)}")
        raise

# Evaluate on all sets
train_r2, train_mse = evaluate_model(X_train, y_train, "Train")
val_r2, val_mse = evaluate_model(X_val, y_val, "Validation")
test_r2, test_mse = evaluate_model(X_test, y_test, "Test")

# ---------------------------
# 5. Optimization with Genetic Algorithm
# ---------------------------
varbound = np.array([
    [2, 8],      # NaOH (%)
    [30, 90],    # Time (min)
    [10, 20]     # SSR
])

def ann_objective(X):
    X_scaled = scaler_X.transform([X])
    y_pred = model.predict(X_scaled, verbose=0)
    retention, delignification = scaler_y.inverse_transform(y_pred)[0]
    
    if retention < 85:  # Constraint for minimum retention
        return 1e6
    
    # Maximize both retention and delignification (60-40 weight)
    return -(0.6*retention + 0.4*delignification)

algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 20,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 10
}

# Run optimization
print("\nStarting Genetic Algorithm optimization...")
ga_model = ga(function=ann_objective,
             dimension=3,
             variable_type='real',
             variable_boundaries=varbound,
             algorithm_parameters=algorithm_param)

ga_model.run()

# Get and display optimization results
optimal_params = ga_model.output_dict['variable']
optimal_scaled = scaler_X.transform([optimal_params])
final_pred = scaler_y.inverse_transform(model.predict(optimal_scaled, verbose=0))[0]

print("\n=== Optimization Results ===")
print(f"Best fitness (objective value): {ga_model.output_dict['function']}")
print("\n=== Optimal Process Parameters ===")
print(f"NaOH Concentration: {optimal_params[0]:.2f}%")
print(f"Time: {optimal_params[1]:.2f} min")
print(f"SSR: {optimal_params[2]:.2f}")
print("\n=== Predicted Outcomes ===")
print(f"Predicted Retention: {final_pred[0]:.2f}%")
print(f"Predicted Delignification: {final_pred[1]:.2f}%")

# ---------------------------
# 6. Results & Visualization
# ---------------------------
# Generate predictions for optimal parameters
optimal_scaled = scaler_X.transform([optimal_params])
final_pred = scaler_y.inverse_transform(model.predict(optimal_scaled, verbose=0))[0]

print("\n=== Optimal Conditions ===")
print(f"NaOH Concentration: {optimal_params[0]:.2f}%")
print(f"Time: {optimal_params[1]:.2f} min")
print(f"SSR: {optimal_params[2]:.2f}")
print(f"Predicted Retention: {final_pred[0]:.2f}%")
print(f"Predicted Delignification: {final_pred[1]:.2f}%")

# Plot learning curve with lower DPI
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('ANN Training Progress')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)  # Reduced DPI
plt.close()

# 3D Surface plots with reduced resolution
n = 20  # Further reduced resolution
ssr_fixed = np.median(df_rh['ssr (solid:solvent)'])
naoh_range = np.linspace(varbound[0, 0], varbound[0, 1], n)
time_range = np.linspace(varbound[1, 0], varbound[1, 1], n)
NaOH, Time = np.meshgrid(naoh_range, time_range)
X_grid = np.column_stack([NaOH.ravel(), Time.ravel(), np.full(NaOH.size, ssr_fixed)])
X_grid_scaled = scaler_X.transform(X_grid)
y_grid = scaler_y.inverse_transform(model.predict(X_grid_scaled, verbose=0))
Retention = y_grid[:, 0].reshape(n, n)
Delignification = y_grid[:, 1].reshape(n, n)

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(NaOH, Time, Retention, cmap='viridis')
ax1.set_xlabel('NaOH (%)')
ax1.set_ylabel('Time (min)')
ax1.set_zlabel('Retention (%)')
ax1.set_title(f'Retention Surface (SSR={ssr_fixed:.2f})')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(NaOH, Time, Delignification, cmap='plasma')
ax2.set_xlabel('NaOH (%)')
ax2.set_ylabel('Time (min)')
ax2.set_zlabel('Delignification (%)')
ax2.set_title(f'Delignification Surface (SSR={ssr_fixed:.2f})')

plt.tight_layout()
plt.savefig('response_surfaces.png', dpi=150)
plt.close()

# Cleanup
try:
    shutil.rmtree(temp_dir)
except Exception as e:
    print(f"Warning: Could not remove temporary directory: {str(e)}")

# Final garbage collection
gc.collect()

print("\nAll results and plots saved. Script completed successfully.")
