# ANN Optimization for Biomass Pretreatment

## Description
Neural network model to optimize rice husk alkaline pretreatment conditions using TensorFlow and genetic algorithm.

## Features
- Data preprocessing and augmentation
- ANN model with LeakyReLU activation
- Genetic algorithm optimization
- 3D visualization of results
- Automated parameter tuning

## Requirements
```bash
tensorflow>=2.8.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
geneticalgorithm>=1.0.2
```

## Usage
1. Prepare input data in 'experiment datafor model.xlsx' with columns:
   - biomass_code
   - ssr (solid:solvent)
   - naoh (%)
   - time (min)
   - retention (%)
   - delignification (%)

2. Run the script:
```bash
python improved_ann.py
```

## Output
- learning_curve.png: Training progress visualization
- response_surfaces.png: 3D visualization of model predictions
- Optimal process parameters in console output

## Project Structure
```
├── improved_ann.py      # Main script
├── requirements.txt     # Dependencies
├── README.md           # Documentation
└── .gitignore         # Git ignore rules
```
## For Collaborators to Clone and Work
```
git clone https://github.com/P-MUNDA/ann.git
cd ann
pip install -r requirements.txt
```

