import numpy as np
import os
from src.data_preprocess import load_and_preprocess_data
from src.model_train import create_model, train_model
from src.evaluate import plot_history
from src.utils import save_model, plot_confusion_matrix

def main():
    # Ensure necessary directories exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')

    # Load and preprocess data
    (X_train, y_train), (X_test, y_test), datagen = load_and_preprocess_data()
    
    # Create and train the model
    model = create_model()
    history = train_model(model, datagen, X_train, y_train, X_test, y_test, 'models/image_classification_model.h5')
    
    # Plot training results
    plot_history(history, 'results')
    
    # Generate predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, classes=[str(i) for i in range(10)], results_dir='results')

if __name__ == "__main__":
    main()
