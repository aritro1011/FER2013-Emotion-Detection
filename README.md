Sure! Here's a `README.md` for your CNN emotion detection project:

---

# CNN Emotion Detection Project

This project utilizes a Convolutional Neural Network (CNN) to detect emotions from facial expressions. It classifies emotions such as happiness, sadness, anger, surprise, and fear. The model is trained on a customized version of the FER2013 dataset. The goal is to build a robust emotion recognition system for use in real-time applications.

## Custom Dataset Modifications

For this project, the FER2013 dataset has been customized by:

1. **Removing the "disgust" emotion**: The disgust category is excluded to improve focus on the other emotions.
2. **Equalizing the image count**: All remaining emotions (happiness, sadness, anger, surprise, and fear) have been set to 3,171 images each to balance the dataset and avoid bias.

## Requirements

- Python 3.x
- TensorFlow (or Keras)
- OpenCV
- NumPy
- Matplotlib
- FER2013 dataset (customized as mentioned above)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/emotion-detection.git
    cd emotion-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the customized FER2013 dataset and place it in the `data/` directory.

## Training the Model

Run the training script to start the training process:
```bash
python train_model.py
```

The model will be trained on the customized dataset and saved as `emotion_model.h5`.

## Plan to Convert into a Chrome Extension

In order to integrate the emotion detection system as a Chrome extension, the following steps are planned:

1. **Frontend Development (HTML/CSS/JS)**:
    - Create a simple user interface (UI) that can interact with the model.
    - Use JavaScript to capture webcam video from the browser.

2. **Backend Model Integration**:
    - Convert the trained CNN model into a TensorFlow.js format.
    - Use TensorFlow.js to load and run the emotion detection model directly within the browser.

3. **Chrome Extension Setup**:
    - Develop the Chrome extension using the `manifest.json` file to define the extension's behavior.
    - Embed the frontend UI and JavaScript logic to interact with the webcam and display detected emotions.

4. **Real-time Emotion Detection**:
    - Capture webcam feed in real-time, process each frame, and detect the emotion using the converted TensorFlow.js model.
    - Display the predicted emotion as an overlay on the webpage.

5. **Testing and Optimization**:
    - Perform thorough testing to ensure the extension works smoothly across different devices.
    - Optimize the model for speed and accuracy within the constraints of a browser environment.

## Usage

Once the model is trained, it can be used for emotion detection through the following script:
```bash
python detect_emotion.py --image <path_to_image>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This `README.md` should guide users through setting up the project and give them insight into how to use it, along with plans for the future Chrome extension.
