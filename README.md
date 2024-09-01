
# FutureBaby - Predict Your Future Child's Appearance

FutureBaby is a simple Python application that allows you to predict what your future child might look like by blending the facial features of two parent images.

## Features

- Upload photos of two parents.
- Automatically detect facial landmarks.
- Blend the facial features of the parents to generate an image of the potential child.
- Display and save the resulting image.

## Requirements

To run this project, you will need to have the following installed:

- Python 3.x
- OpenCV
- dlib
- NumPy
- PIL (Python Imaging Library)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/futurebaby.git
   cd futurebaby
   ```

2. **Install dependencies**:

   You can install the required Python libraries using pip:

   ```bash
   pip install opencv-python dlib numpy pillow
   ```

3. **Download the shape predictor model**:

   Download the `shape_predictor_68_face_landmarks.dat` file from the official dlib repository or via the following link:

   [Download shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

   After downloading, extract the `.dat` file and place it in the project directory.

## Usage

1. **Prepare the parent images**:

   Ensure that you have two images of the parents (e.g., `parent1.jpg` and `parent2.jpg`) in the project directory.

2. **Run the script**:

   ```bash
   python futurebaby.py
   ```

3. **View the result**:

   The generated image of the future child will be saved as `future_baby.jpg` in the project directory. The image will also be displayed using the default image viewer.

## Example

Here's how you can use the application:

```python
import cv2
import dlib
import numpy as np
from PIL import Image

# Load the facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to get landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) > 0:
        return np.array([[p.x, p.y] for p in predictor(gray, rects[0]).parts()])
    else:
        return None

# Function to blend faces
def blend_faces(img1, img2):
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)

    if landmarks1 is None or landmarks2 is None:
        print("Could not find faces in one of the images.")
        return None

    blended_landmarks = (landmarks1 + landmarks2) / 2
    blended_face = np.zeros_like(img1)
    
    for i in range(0, len(blended_landmarks)):
        cv2.circle(blended_face, (int(blended_landmarks[i][0]), int(blended_landmarks[i][1])), 2, (255, 255, 255), -1)

    return blended_face

# Load parent images
image1 = cv2.imread("parent1.jpg")
image2 = cv2.imread("parent2.jpg")

# Generate future baby face
child_face = blend_faces(image1, image2)

# Save and display the result
if child_face is not None:
    cv2.imwrite("future_baby.jpg", child_face)
    Image.open("future_baby.jpg").show()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is a simple demonstration and is not intended to be used for serious predictions. The results are purely speculative and should not be taken as accurate representations of what a future child might actually look like.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any inquiries, please contact [yourname@domain.com].
```

### Explanation:

- **Overview**: Provides a high-level summary of what the project does.
- **Features**: Lists the primary functionalities of the application.
- **Requirements**: Details the necessary software dependencies.
- **Installation**: Guides users on how to install the project.
- **Usage**: Explains how to use the application.
- **Example**: Provides an example script that the user can run.
- **License**: Indicates the licensing for the project.
- **Disclaimer**: Mentions the speculative nature of the results.
- **Contributing**: Encourages contributions from others.
- **Contact**: Provides a way for users to reach out.

This `README.md` file should give users all the information they need to understand, install, and use your project.
