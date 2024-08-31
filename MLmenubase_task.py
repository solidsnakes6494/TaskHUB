import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Function to automatically preprocess a dataset
def auto_preprocess(data, target_column=None):
    if target_column:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        X = data.copy()
        y = None

    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object', 'category']).columns

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

    if target_column:
        return X_processed, y
    else:
        return X_processed

# Function to capture an image using the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        print("Failed to capture image")
        return None

# Function to detect and crop the face
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No face detected")
        return image, None

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        return image, face

    return image, None

# Function to display the original image with the cropped face using matplotlib
def show_image_with_face(original, face):
    if face is not None:
        face_resized = cv2.resize(face, (150, 150))
        original[0:150, 0:150] = face_resized

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.imshow(original_rgb)
    plt.axis('off')
    plt.show()

# Function to load the image
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to load image from {path}")
    return image

# Function to apply different filters to the image
def apply_filters(image):
    filters = {
        "Original": image,
        "Grayscale": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "Blur (Gaussian)": cv2.GaussianBlur(image, (15, 15), 0),
        "Edge Detection (Canny)": cv2.Canny(image, 100, 200),
        "Sharpen": cv2.filter2D(image, -1, 
                                kernel=np.array([[0, -1, 0], 
                                                 [-1, 5, -1], 
                                                 [0, -1, 0]])),
        "Emboss": cv2.filter2D(image, -1, 
                               kernel=np.array([[-2, -1, 0], 
                                                [-1, 1, 1], 
                                                [0, 1, 2]])),
        "Sepia": cv2.transform(image, 
                               np.matrix([[0.272, 0.534, 0.131], 
                                          [0.349, 0.686, 0.168], 
                                          [0.393, 0.769, 0.189]])),
    }
    return filters

# Function to display images with filters
def display_images(filters):
    num_filters = len(filters)
    plt.figure(figsize=(12, 8))

    for i, (filter_name, filtered_image) in enumerate(filters.items(), 1):
        plt.subplot(2, (num_filters + 1) // 2, i)
        if len(filtered_image.shape) == 2:  # Grayscale image
            plt.imshow(filtered_image, cmap='gray')
        else:  # BGR image
            plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.title(filter_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to create and display a custom image with gradients
def create_custom_image():
    height, width = 100, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)

    red = [255, 0, 0]
    green = [0, 255, 0]
    blue = [0, 0, 255]
    yellow = [255, 255, 0]

    image[:50, :50] = red
    image[:50, 50:] = green
    image[50:, :50] = blue
    image[50:, 50:] = yellow

    for i in range(height):
        image[i, :, 0] = np.linspace(0, 255, width)
        image[:, i, 2] = np.linspace(255, 0, height)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Menu-based interface
def menu():
    while True:
        print("\nMenu:")
        print("1. Auto Preprocess a Dataset")
        print("2. Capture and Display Image with Face Detection")
        print("3. Apply Filters to an Image")
        print("4. Create and Display a Custom Image with Gradients")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            file_path = input("Enter the dataset file path (CSV format): ")
            df = pd.read_csv(file_path)
            target_column = input("Enter the target column name (or press Enter if none): ")
            if target_column == '':
                target_column = None
            X_processed, y = auto_preprocess(df, target_column=target_column)
            print("Preprocessing Complete!")
            print(X_processed.head())

        elif choice == '2':
            image = capture_image()
            if image is not None:
                original_image, face_cropped = detect_and_crop_face(image)
                show_image_with_face(original_image, face_cropped)

        elif choice == '3':
            image_path = input("Enter the image file path: ")
            image = load_image(image_path)
            if image is not None:
                filters = apply_filters(image)
                display_images(filters)

        elif choice == '4':
            create_custom_image()

        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
