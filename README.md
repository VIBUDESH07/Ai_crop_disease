

# AI Crop Disease Detection

This project leverages AI and machine learning to detect crop diseases from images. By using a deep learning model, farmers and agricultural experts can identify potential diseases in crops quickly and efficiently, reducing the impact of diseases on crop yields.

## Key Features
- **Real-time Disease Detection**: Identifies diseases in crop images through a trained AI model.
- **Multi-Crop Support**: Supports detection of diseases in multiple crop types such as wheat, corn, rice, and more.
- **High Accuracy**: Utilizes state-of-the-art deep learning techniques for accurate disease detection.
- **User-Friendly Interface**: Provides an intuitive interface for uploading crop images and viewing results.
- **Offline Mode**: Allows the model to be deployed on mobile devices or edge devices for use in remote areas without internet access.

## Technology Stack
- **Frontend**: React.js (for user interaction and image upload)
- **Backend**: Flask (handles the AI inference requests and model serving)
- **Machine Learning Framework**: TensorFlow / PyTorch (used for model training and inference)
- **Database**: MongoDB (stores user data, disease history, etc.)
- **Deployment**: Docker, AWS EC2

## Project Architecture
1. **Image Upload**: Users can upload images of crops through the frontend (React.js).
2. **Preprocessing**: The image is preprocessed (resized, normalized) to match the model's input format.
3. **Disease Prediction**: The preprocessed image is passed to the backend, where the trained model predicts the type of disease.
4. **Result Display**: The disease prediction and suggested treatment are displayed on the frontend.
5. **User Data Storage**: If the user is registered, their previous crop submissions and diagnoses are stored in MongoDB.

## How It Works
1. **Collect Crop Images**: The system gathers labeled images of diseased and healthy crops.
2. **Train Model**: A convolutional neural network (CNN) is trained on these labeled images to learn patterns of diseases in the crops.
3. **Deploy Model**: The trained model is deployed via Flask API, where it accepts user-uploaded images and returns predictions.
4. **Disease Identification**: The model outputs the type of disease affecting the crop, if any, along with a confidence score.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow or PyTorch
- Flask
- React.js
- MongoDB

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crop-disease-detection.git
   ```
2. Install the required dependencies for the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```
4. Train or load the pre-trained model and place it in the `models` directory.
5. Run the backend:
   ```bash
   python app.py
   ```
6. Start the frontend:
   ```bash
   npm start
   ```
7. Access the application at `http://localhost:3000`.

## Dataset
The model is trained using the **PlantVillage dataset**, which contains thousands of labeled images of crops affected by diseases. You can download the dataset from [Kaggle](https://www.kaggle.com/emmarex/plantdisease).

## Model Details
- **Architecture**: Convolutional Neural Networks (ResNet50, EfficientNet)
- **Input**: 224x224 RGB image of a crop
- **Output**: Predicted disease class (e.g., late blight, leaf spot, etc.)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Future Enhancements
- **Multi-Language Support**: Add support for local languages to help farmers in different regions.
- **Mobile App**: Develop a mobile application for ease of use in the field.
- **Additional Crop Support**: Extend the model to support more crops and diseases.
- **Cloud Integration**: Deploy the model on the cloud for scalability 
