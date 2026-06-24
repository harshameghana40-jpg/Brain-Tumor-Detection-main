# 🧠 Brain Tumor Detection AI

A comprehensive, locally deployable brain tumor detection system using TensorFlow, Flask, and modern web technologies. This system can classify brain MRI scans into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor.

## ✨ Features

- **🤖 AI-Powered Detection**: Uses transfer learning with MobileNetV2 for accurate tumor classification
- **🌐 Web Interface**: Beautiful, responsive web UI with drag-and-drop file upload
- **🌙 Dark/Light Mode**: Toggle between themes with system preference detection
- **📊 Real-time Results**: Instant analysis with confidence scores and clinical information
- **💻 Local Deployment**: Completely offline-capable system
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🔒 Privacy-First**: All processing happens locally, no data sent to external servers

## 📋 Requirements

- Python 3.8 or higher
- 8GB+ RAM (recommended for model training)
- GPU support (optional, for faster training)
- Brain Tumor MRI Dataset from Kaggle

## 🚀 Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Download the Brain Tumor MRI dataset from Kaggle and extract it to the project folder:

```
Brain Tumor Detection-AI/
├── dataset/
│   ├── train/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── pituitary_tumor/
│   │   └── no_tumor/
│   └── test/
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── pituitary_tumor/
│       └── no_tumor/
```

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a CNN model using transfer learning
- Generate training metrics and confusion matrix
- Save the trained model to `model/brain_model.h5`

### 5. Run the Application

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

## 📁 Project Structure

```
Brain Tumor Detection-AI/
├── app.py                  # Flask backend application
├── train_model.py          # Model training script
├── tumor_info.json         # Clinical metadata for tumor types
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model/
│   └── brain_model.h5      # Trained TensorFlow model
├── dataset/               # Your Kaggle dataset goes here
│   ├── train/
│   └── test/
├── static/
│   ├── css/
│   │   └── style.css      # Custom styles with dark/light mode
│   ├── js/
│   │   └── script.js      # Frontend JavaScript
│   └── uploads/           # Temporary upload storage
└── templates/
    └── index.html         # Main web interface
```

## 🎯 Usage

### Web Interface

1. **Upload Image**: Drag and drop or click to upload a brain MRI scan
2. **Analyze**: Click "Analyze MRI" to process the image
3. **View Results**: See detailed analysis including:
   - Tumor type classification
   - Confidence score
   - Symptoms
   - Treatment recommendations
   - Cure rate
   - Severity level

### API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Upload and analyze image
- `GET /health` - System health check
- `GET /uploads/<filename>` - Serve uploaded files

### Example API Response

```json
{
  "tumor_type": "Glioma Tumor",
  "confidence": "96.8%",
  "symptoms": "Seizures, headaches, memory loss, personality changes",
  "treatment": "Surgery, Radiation Therapy, Chemotherapy, Targeted Therapy",
  "cure_rate": "65%",
  "severity": "High",
  "description": "Gliomas are tumors that develop from glial cells...",
  "color": "#dc3545",
  "class_name": "glioma_tumor"
}
```

## 🎨 Features

### Dark/Light Mode
- Toggle between themes using the button in the navigation
- Auto mode detects system preference
- Theme preference is saved locally

### Responsive Design
- Works on all device sizes
- Touch-friendly interface
- Optimized for mobile use

### File Support
- **Formats**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Size Limit**: 16MB maximum
- **Drag & Drop**: Easy file upload

### Keyboard Shortcuts
- `Ctrl/Cmd + U`: Upload file
- `Ctrl/Cmd + Enter`: Analyze image
- `Ctrl/Cmd + T`: Toggle theme

## 🔧 Configuration

### Model Training Parameters

Edit `train_model.py` to customize:

```python
self.img_size = (224, 224)      # Image size
self.batch_size = 32            # Batch size
self.epochs = 20                # Training epochs
self.learning_rate = 0.001      # Learning rate
```

### Tumor Information

Update `tumor_info.json` to modify clinical information:

```json
{
  "glioma_tumor": {
    "name": "Glioma Tumor",
    "symptoms": "Seizures, headaches, memory loss",
    "treatment": "Surgery, Radiation, Chemotherapy",
    "cure_rate": "65%",
    "severity": "High",
    "description": "Description here...",
    "color": "#dc3545"
  }
}
```

## 📊 Model Performance

The system uses:
- **Architecture**: MobileNetV2 with transfer learning
- **Input Size**: 224x224 pixels
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Data Augmentation**: Rotation, zoom, flip, shear
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout layers

## 🛠️ Development

### Adding New Tumor Types

1. Add new class to dataset folders
2. Update `tumor_info.json` with clinical data
3. Retrain the model
4. Update frontend display logic

### Customizing the UI

- **CSS**: Modify `static/css/style.css`
- **JavaScript**: Edit `static/js/script.js`
- **HTML**: Update `templates/index.html`

### API Extensions

Add new endpoints in `app.py`:

```python
@app.route('/api/new-endpoint', methods=['GET'])
def new_endpoint():
    return jsonify({"message": "New endpoint"})
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure `model/brain_model.h5` exists
2. **Dataset error**: Check dataset folder structure
3. **Memory issues**: Reduce batch size in training
4. **Port already in use**: Change port in `app.py`

### Performance Tips

- Use GPU for faster training
- Reduce image size for faster inference
- Use SSD storage for better I/O performance
- Increase RAM for larger batch sizes

## 📝 License

This project is for educational and research purposes. Always consult healthcare professionals for medical diagnosis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue on GitHub

## 🔮 Future Enhancements

- [ ] Support for DICOM files
- [ ] Batch processing
- [ ] Model ensemble
- [ ] Real-time video analysis
- [ ] Mobile app version
- [ ] Cloud deployment options

---

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical decisions. 