import torch
import cv2
import numpy as np

#############################################
# FUNCTIONS
#############################################

# PREPROCESSING
# --------------------------------------------

# function to check if image is black and white

def is_color(img):
    # check if image has 3 channels
    dim = len(np.array(img).shape)
    
    if dim == 2: # with 1 channel dimension = (224,224) => greyscale image
        return 0
    # if image has 3 channels and all channels are equal
    elif dim == 3: # for example (224, 224, 3)
        if ((np.array(img)[:,:,2] == np.array(img)[:,:,1]).all() == True &
            (np.array(img)[:,:,0] == np.array(img)[:,:,1]).all() == True &
            (np.array(img)[:,:,0] == np.array(img)[:,:,2]).all() == True ):
            return 0
        else:
            return 1
    else:
        return 0

def preprocess_df(df, image_dir, process_mediapipe=False):
    """
    Preprocesses a DataFrame by loading and analyzing corresponding face images.

    This function performs the following steps:
    - Adds default values for missing 'gender' and 'FaceOcclusion' columns (for test data).
    - Drops rows with missing FaceOcclusion.
    - Adds metadata columns such as gender_id, database number, and count.
    - Initializes and fills image-related properties for each image:
      - Dimensions (width, height), number of channels, number of pixels
      - Mean and standard deviation of pixel values (global and by color channel)
      - Grayscale vs color detection
    - Optionally initializes columns for MediaPipe post-processing (if process_mediapipe=True).

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame, must include a 'filename' column pointing to image files.

    image_dir : str
        Path to the directory where images are stored.

    process_mediapipe : bool, optional (default=False)
        If True, initializes placeholder columns for MediaPipe output (face mask, pixels, etc.).

    Returns:
    --------
    df : pandas.DataFrame
        The preprocessed DataFrame with additional image-related features.
    """
    df = df.copy()  # ← Évite les SettingWithCopyWarning

    # Add missing columns for test sets
    if 'gender' not in df.columns:
        df['gender'] = -1.0
    if 'FaceOcclusion' not in df.columns:
        df['FaceOcclusion'] = -1.0

    # Preprocessing
    df['initial_index'] = df.index
    df = df.dropna(subset=['FaceOcclusion']).copy()

    df.loc[:, 'gender_id'] = np.round(df['gender']).astype(int)
    df.loc[:, 'db_number'] = df['filename'].apply(lambda x: (x.split('/')[0])[-1]).astype(int)
    df.loc[:, 'count'] = 1

    # Initialize image properties
    for col in [
        'color', 'image_width', 'image_height', 'channels', 'pixels',
        'pixels_mean', 'pixels_mean_R', 'pixels_mean_G', 'pixels_mean_B',
        'pixels_std', 'pixels_std_R', 'pixels_std_G', 'pixels_std_B'
    ]:
        df.loc[:, col] = 0

    if process_mediapipe:
        df.loc[:, 'face'] = 1
        df.loc[:, 'face_pixels'] = 0
        df.loc[:, 'mask_pixels'] = 0
        print('Use preprocessing_mediapipe.py to get mediapipe mask, mesh and contours')

    # Loop through images
    for i in df.index:
        if i % 5000 == 0:
            print(i)

        try:
            filename = df.loc[i, 'filename']
            image_path = f"{image_dir}/{filename}"
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError("Image not found or unreadable")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            df.loc[i, 'image_width'] = image.shape[0]
            df.loc[i, 'image_height'] = image.shape[1]
            df.loc[i, 'pixels'] = image.shape[0] * image.shape[1]
            df.loc[i, 'channels'] = image.shape[2]
            df.loc[i, 'pixels_mean'] = np.mean(image)
            df.loc[i, 'pixels_mean_R'] = np.mean(image[:, :, 0])
            df.loc[i, 'pixels_mean_G'] = np.mean(image[:, :, 1])
            df.loc[i, 'pixels_mean_B'] = np.mean(image[:, :, 2])
            df.loc[i, 'pixels_std'] = np.std(image)
            df.loc[i, 'pixels_std_R'] = np.std(image[:, :, 0])
            df.loc[i, 'pixels_std_G'] = np.std(image[:, :, 1])
            df.loc[i, 'pixels_std_B'] = np.std(image[:, :, 2])

            if is_color(image) == 1:
                df.loc[i, 'color'] = 1

        except Exception as e:
            print(f"Could not process {filename} (index {i}): {e}")

    df.loc[:, 'no_color'] = 1 - df['color']

    return df

def process_mediapipe(df, image_dir, save_to_file=False):
    '''function to apply mediapipe face mesh model to images, get masked images, 
    contours and mesh, and return 468 landmarks (3D keypoints)
    Input :
        df : panda dataframe with image filenames
        image_dir : directory where images are stored
    Output :
        df : panda dataframe with additional columns for number of faces detected, face pixels, mask pixels,
        masked_image: image with skin area, ans black mask overlay (ovale) elsewhere 
        contours : image with face ovale contours, brows and lips in green
        mesh : image with face mesh keypoints
        landmarks : 468 landmarks (3D keypoints) in mediapipe format (madiapipe array of (x, y, z) coordinates in dictionary format)
    '''
#---------------------------------------
# SHOW IMAGES
#---------------------------------------

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

#---------------------------------------
# DEFINE DEVICE
#---------------------------------------
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

#---------------------------------------
# DEFINE DATASET CLASS
#---------------------------------------

#class Dataset(torch.utils.data.Dataset): # in "starter notebook code" dataset is imported from torch.utils.data
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, image_dir):
         'Initialization'
         self.image_dir = image_dir
         self.df = df
         self.transform = transforms.ToTensor()
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.loc[index]
        filename = row['filename']

        # Load data and get label
        img = Image.open(f"{self.image_dir}/{filename}")  
        X = self.transform(img)

        y = row['FaceOcclusion']     
        y = np.float32(y)

        gender = row['gender_id'] # changed to have round values 0 or 1

        return X, y, gender, filename

#--------------------------------------------------
# Compute mean and standard deviation on the pixels
#--------------------------------------------------

def calculate_mean_std(loader, num_channels=3):
    '''Calculate mean and standard deviation of the dataset.
    Args:
        loader: DataLoader object
        num_channels: number of channels
    Returns:
        mean: mean of the dataset (tensor)
        std: standard deviation of the dataset (tensor)
    '''
    device = get_device()
    channel_sum = torch.zeros(num_channels).to(device)
    channel_squared_sum = torch.zeros(num_channels).to(device)
    num_elements = 0

    for data, _, _, _ in loader:
        data = data.to(device)
        channel_sum += data.sum(dim=[0, 2, 3])
        channel_squared_sum += (data ** 2).sum(dim=[0, 2, 3])
        num_elements += data.size(0) * data.size(2) * data.size(3)

    mean = channel_sum / num_elements
    std = (channel_squared_sum / num_elements - mean ** 2) ** 0.5
    return mean, std