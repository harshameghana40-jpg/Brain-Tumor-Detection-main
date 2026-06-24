import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class BrainTumorDetector:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 20
        self.learning_rate = 0.001
        self.model = None
        self.class_names = []
        self.history = None
        
    def check_dataset(self):
        """Check if dataset folders exist and get class names"""
        train_path = './dataset/train'
        test_path = './dataset/test'
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print("❌ Dataset folders not found!")
            print("Please make sure `./dataset/train/` and `./dataset/test/` exist in your project folder before continuing.")
            print("\nExpected structure:")
            print("dataset/")
            print("├── train/")
            print("│   ├── glioma_tumor/")
            print("│   ├── meningioma_tumor/")
            print("│   ├── pituitary_tumor/")
            print("│   └── no_tumor/")
            print("└── test/")
            print("    ├── glioma_tumor/")
            print("    ├── meningioma_tumor/")
            print("    ├── pituitary_tumor/")
            print("    └── no_tumor/")
            return False
            
        # Get class names from train folder
        self.class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
        print(f"✅ Dataset found! Classes: {self.class_names}")
        return True
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            './dataset/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            './dataset/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        print(f"✅ Data generators created!")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
    
    def build_model(self):
        """Build the CNN model using transfer learning"""
        print("🔨 Building model...")
        
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create the model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        self.model.summary()
    
    def train_model(self):
        """Train the model"""
        print("🚀 Starting training...")
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                'model/brain_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.test_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
    
    def evaluate_model(self):
        """Evaluate the model and print metrics"""
        print("📊 Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Print classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('model', exist_ok=True)
        self.model.save('model/brain_model.h5')
        print("✅ Model saved to model/brain_model.h5")
    
    def run_training(self):
        """Run the complete training pipeline"""
        print("🧠 Brain Tumor Detection Model Training")
        print("=" * 50)
        
        # Check dataset
        if not self.check_dataset():
            return
        
        # Create data generators
        self.create_data_generators()
        
        # Build model
        self.build_model()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        self.evaluate_model()
        
        # Plot history
        self.plot_training_history()
        
        # Save model
        self.save_model()
        
        print("\n🎉 Training pipeline completed successfully!")
        print("📁 Model saved to: model/brain_model.h5")
        print("📊 Metrics saved to: confusion_matrix.png, training_history.png")

if __name__ == "__main__":
    detector = BrainTumorDetector()
    detector.run_training() 