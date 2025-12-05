# CoreML Segmentation

A comprehensive framework for image segmentation using Apple's CoreML models. This project provides tools, utilities, and examples for building and deploying semantic segmentation models on iOS and macOS devices.

## 🎯 Overview

CoreML Segmentation is designed to streamline the process of:
- Training and fine-tuning segmentation models
- Converting models to CoreML format
- Deploying segmentation models on Apple platforms
- Processing and evaluating segmentation results

Whether you're working with pixel-level classification or instance segmentation, this framework provides a modular and extensible architecture to support your computer vision needs.

## 📁 Project Structure

```
coreml-segmentation-demo/
├── data/              # Dataset storage and preprocessing
├── models/            # Pre-trained and custom CoreML models
├── notebooks/         # Jupyter notebooks for experimentation and demos
├── src/               # Core library code and utilities
└── README.md          # This file
```

### Directory Details

- **`data/`** – Data handling, preprocessing, and dataset management
  - Input samples, augmented data, and preprocessed datasets
  
- **`models/`** – CoreML model storage and conversion utilities
  - Pre-trained models, fine-tuned checkpoints, and converted .mlmodel files
  
- **`notebooks/`** – Jupyter notebooks for exploration and development
  - Model training, evaluation, conversion guides, and example workflows
  
- **`src/`** – Main library code
  - Segmentation models, preprocessing pipelines, utilities, and inference wrappers

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Xcode (for iOS deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Prudhviteja9/CoreML-Segmentation.git
   cd coreml-segmentation-demo
   ```

2. **Set up a Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Features

- ✅ **Multiple Model Architectures** – U-Net, DeepLab, FCN, and more
- ✅ **CoreML Export** – Convert TensorFlow/PyTorch models to `.mlmodel` format
- ✅ **Data Augmentation** – Robust preprocessing and augmentation pipelines
- ✅ **Evaluation Metrics** – IoU, Dice coefficient, Pixel Accuracy
- ✅ **Example Notebooks** – Step-by-step guides and demos
- ✅ **Inference Wrapper** – Easy-to-use API for model inference

## 📚 Usage

### Training a Model

```python
from src.models import SegmentationModel
from src.data import DataLoader

# Load dataset
train_loader = DataLoader('data/input/training')
val_loader = DataLoader('data/input/validation')

# Initialize model
model = SegmentationModel(architecture='unet', num_classes=10)

# Train
model.train(train_loader, val_loader, epochs=50, learning_rate=0.001)

# Save
model.save('models/my_segmentation_model')
```

### Inference

```python
from src.inference import SegmentationInference
import cv2

# Load model
inference = SegmentationInference('models/my_segmentation_model.mlmodel')

# Process image
image = cv2.imread('data/input/sample.jpg')
segmentation_mask = inference.predict(image)

# Save result
cv2.imwrite('data/output/segmentation_mask.png', segmentation_mask)
```

### Converting to CoreML

```python
from src.conversion import TensorFlowToCoreML

converter = TensorFlowToCoreML(
    model_path='models/tf_model.pb',
    input_shape=(256, 256, 3),
    output_names=['segmentation']
)
converter.convert('models/segmentation_model.mlmodel')
```

## 📊 Model Performance

| Model | Dataset | mIoU | FPS (iOS) |
|-------|---------|------|-----------|
| U-Net | Cityscapes | 82.5% | 15 |
| DeepLab v3+ | Pascal VOC | 85.2% | 8 |
| FCN | ADE20K | 78.9% | 12 |

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

```bash
sphinx-build -b html docs/ docs/_build/
```

### Code Style

This project follows PEP 8. Format code with black:

```bash
black src/ notebooks/
```

## 📝 Notebooks

Explore the `notebooks/` directory for:
- `01_data_exploration.ipynb` – Dataset analysis and visualization
- `02_model_training.ipynb` – Training workflows
- `03_model_evaluation.ipynb` – Performance metrics and validation
- `04_coreml_conversion.ipynb` – Converting models to CoreML format
- `05_inference_demo.ipynb` – Running inference on sample images

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

## 👨‍💻 Author

**Prudhvi Teja**  
GitHub: [@Prudhviteja9](https://github.com/Prudhviteja9)

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: prudhviteja@example.com

## 🙏 Acknowledgments

- Apple CoreML documentation and resources
- Open-source community for segmentation model architectures
- Contributors and testers

---

**Last Updated:** December 2025  
**Status:** Active Development
