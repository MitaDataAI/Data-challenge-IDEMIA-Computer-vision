import numpy as np                    
import matplotlib.pyplot as plt       
from PIL import Image                 
import cv2                            # Used for image processing (mask creation, bitwise operations)
import mediapipe as mp                # Used for Face Mesh landmark detection

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------
# Display Utilities
# --------------------------------------------------

def show_images(df, image_dir, n_images=5, title='None'):
    N = len(df)
    plt.figure(figsize=(15, (N//5 + 1)*3))
    for i in range(N):
        plt.subplot(N//5 + 1, n_images, i + 1)
        img = Image.open(f"{image_dir}/{df.iloc[i]['filename']}")
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


# --------------------------------------------------
# Check if Image is Black and White
# --------------------------------------------------

def is_black_and_white(img):
    img_np = np.array(img)
    dim = len(img_np.shape)

    if dim == 2:
        return True
    elif dim == 3:
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        return (r == g).all() and (r == b).all()
    return False


# --------------------------------------------------
# Extract Landmark Points from Face Mesh
# --------------------------------------------------

def get_points(image, results):
    skin_landmark_indices = list(range(0, 468))
    points = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in skin_landmark_indices:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                points.append([x, y])

    return np.array(points)


# --------------------------------------------------
# Generate Mask from Landmarks
# --------------------------------------------------

def get_mask_from_points(image, points):
    mask = np.zeros_like(image)

    if len(points) > 0:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    else:
        mask = np.ones_like(image) * 255

    return mask


# --------------------------------------------------
# Apply Mask to Image
# --------------------------------------------------

def get_masked_image(image, results):
    points = get_points(image, results)
    mask = get_mask_from_points(image, points)
    masked_image = cv2.bitwise_and(image, mask)
    return mask, masked_image


# --------------------------------------------------
# Get Face Mesh Overlay
# --------------------------------------------------

def get_mesh(image, results):
    meshed_image = image.copy()

    if results.multi_face_landmarks:
        for f, face_landmarks in enumerate(results.multi_face_landmarks):
            if f == 0:
                mp_drawing.draw_landmarks(
                    image=meshed_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
    return meshed_image


# --------------------------------------------------
# Get Face Contour Overlay
# --------------------------------------------------

def get_contours(initial_image, results):
    contoured_image = initial_image.copy()

    if results.multi_face_landmarks:
        for f, face_landmarks in enumerate(results.multi_face_landmarks):
            if f == 0:
                mp_drawing.draw_landmarks(
                    image=contoured_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
    return contoured_image


# --------------------------------------------------
# Display Masking Result
# --------------------------------------------------

def show_mask(image, meshed_image, mask, masked_image, path=''):
    plt.figure(figsize=(12, 3.5))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 4, 2)
    plt.imshow(meshed_image)
    plt.title('Landmarks (Mesh or Contours)')
    plt.subplot(1, 4, 3)
    plt.imshow(mask)
    plt.title('Mask')
    plt.subplot(1, 4, 4)
    plt.imshow(masked_image)
    plt.title('Skin Area')
    plt.suptitle('Skin Area Extraction with MediaPipe Face Mesh Model\n' + path)
    plt.show()
