import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# 🔹 Load the saved best model
model = load_model("emotion_detection_model.keras")

# 🔹 Set test dataset paths
test_dir = 'FER2013/test'
img_size = (48, 48)
batch_size = 32

# 🔹 Define Test Data Generator (No Augmentation, Only Rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# 🔹 Evaluate Model on Test Data
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"❌ Final Test Loss: {test_loss:.4f}\n")

# 🔹 Get Predictions for Confusion Matrix
y_true = test_generator.classes  # Actual labels
y_pred_probs = model.predict(test_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# 🔹 Compute Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
emotion_labels = list(test_generator.class_indices.keys())

# 🔹 Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 🔹 Classification Report
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, target_names=emotion_labels))