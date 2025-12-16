# Client Churn Prediction and Retention using AI

This project predicts customer churn using an **Artificial Neural Network (ANN)** built with **TensorFlow** and **Keras**. The project was developed and run in **Google Colab**.  

## Project Description

The goal is to classify whether a customer will leave the company based on features like credit score, geography, gender, age, balance, and more.  

Data preprocessing includes feature engineering, scaling, and encoding categorical variables. The ANN is trained using early stopping and evaluated on a test set.  

By **hyperparameter tuning and experimentation**, observing accuracy and other metrics, the model achieved a **test accuracy of 0.87**.  

## Dataset

The CSV dataset contains customer information and the target column `Exited` (1 if churned, 0 otherwise).  

## Model Architecture and Methods

- **Model:** Artificial Neural Network (ANN)  
- **Framework:** TensorFlow and Keras  
- **Architecture:**
  ReLU (Rectified Linear Unit) is an activation function that outputs the input directly if it is positive; otherwise, it outputs zero. It helps the network learn non-linear patterns efficiently and reduces the likelihood of the vanishing gradient problem.
  - Input Layer: 12 neurons, **ReLU** activation  
  - Hidden Layer 1: 8 neurons, **ReLU** activation + Dropout 0.2  
  - Hidden Layer 2: 5 neurons, **ReLU** activation + Dropout 0.3  
  - Output Layer: 1 neuron, Sigmoid activation  

- **Optimizer:** Adam with learning rate 0.01  
- **Loss Function:** Binary Crossentropy  
- **Callbacks:** Early stopping based on validation loss

- **Methods Used:**  
  - **Feature Engineering:** One-hot encoding for categorical variables (Geography, Gender)
  - **Data Preprocessing:** Scaling with StandardScaler
  - **Training Techniques:** Hyperparameter tuning, experimentation, early stopping
  - **Evaluation Metrics:** Accuracy, loss curves, predictions on test set

## Usage

1. Mount Google Drive in Colab to access the dataset:  
```python
from google.colab import drive
drive.mount('/content/drive')
dataset = pd.read_csv("/content/drive/MyDrive/Customers_information.csv")

```
2. Run the notebook to:
  - Preprocess the data
  - Train the ANN model
  - Evaluate predictions on the test set
  - Visualize the training history with loss and accuracy plots

## Results

- **Test Accuracy:** 0.87
- Loss and accuracy plotted over epochs
- Predictions on the test set ready for analysis

## Future Improvements

- Further hyperparameter tuning
- Cross-validation
- Try other classifiers (Random Forest, XGBoost)
- Feature importance analysis

---



link
https://github.com/emilyabochkovastudy/Client_Retention_AI/tree/main



