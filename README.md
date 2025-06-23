# Data Challenge 2 with IDEMIA : Computer Vision

## IDEMIA
IDEMIA is the world leader in identity technologies. It specializes in biometric solutions, secure digital identification, and authentication systems for both governments and private sectors.  
![image](https://github.com/user-attachments/assets/30139190-db91-4098-97f9-105a12159ca4)

## By RAKOTONIAINA Pety Ialimita (pety.rakotoniaina@télécom-paris.fr)

## Place during the Data Challenge : Champion! 🏆

---

## Structure and Submission Process
The Data Challenge follows the standard principle of a “Kaggle Competition,” based on real-world data and a specific problem.  
You can download the labeled training data and the test data (without labels) from the Data Challenge website.  
Submit your predictions (in a flat file) on the site; submissions are scored instantly and ranked on a public leaderboard. Multiple submissions are allowed.

---

## Goal
You have 100,000 images of human faces with occlusion labels.  
The challenge is to **predict the percentage of the face that is occluded**.  
The model must perform equally well across **male and female** samples. The gender label is available in the training set.

![image](https://github.com/user-attachments/assets/998844b2-c83d-479a-9834-51ec9970d685)  
Red indicates the occluded (hidden) areas of the face, while blue highlights the visible regions. The final image represents a face with 100% visibility.

---

## Metrics

The evaluation is based on a **Weighted Mean Squared Error (WMSE)**, computed **separately for males and females**, and then combined to produce the final score.

### Step 1: Compute Error per Gender

For each gender group (men and women), the error is calculated as:

$$
\text{Err} = \frac{\sum_{i}{w_i(p_i - GT_i)^2}}{\sum_{i}{w_i}}, \quad w_i = \frac{1}{30} + GT_i
$$

Where:
- \( GT_i \): Ground truth occlusion percentage for sample \( i \)
- \( p_i \): Predicted occlusion percentage for sample \( i \)
- \( w_i \): Sample weight

### Step 2: Final Score Calculation

Once you compute the error for **females** (\( Err_F \)) and **males** (\( Err_M \)), the final leaderboard score is:

$$
\text{Score} = \frac{Err_F + Err_M}{2} + \left| Err_F - Err_M \right|
$$

This formulation encourages models to perform equally well across genders by penalizing imbalanced error.

---

## 📊 Detailed EDA Discoveries & Modeling Rationale

The original dataset contained only three variables: `Filename`, `FaceOcclusion`, and `Gender`. Naturally, our modeling phase began by focusing on the two most informative predictors: **FaceOcclusion** and **Gender**.

---

### 0 – Imbalanced Data & Weighted Loss Function

During the EDA phase (`file 1-2`), we discovered:

- A strong **imbalance in the `FaceOcclusion` variable**
- Different patterns across **genders** in terms of pixel intensity

To mitigate the imbalance, we implemented a **class-weighted loss function**. In imbalanced datasets, models tend to overfit to frequent cases and ignore rare ones. Assigning higher weights to underrepresented classes helps the model learn from them.

#### 🔧 Strategy:
- Applied class weighting for **occlusion**
- Gender imbalance remains partially unaddressed

#### ✅ Results (using ResNet-18):
- **Improved performance** on rare occlusion cases
- **Score dropped from 0.0022 → 0.0017** with weighted loss (file `2-1`)
- **Fairer and more robust** predictions across groups

---

### 1 – Filename-Derived Variables

From the `Filename` field, we engineered **key features** that contributed significantly to the model:

- **Image type** (color or grayscale)
- **Database origin** (DB1, DB2, DB3)
- Metadata used for:
  - Stratified sampling
  - Model routing (color vs grayscale)
  - Subgroup performance analysis

These variables were crucial for customizing training and improving generalization.

---

### 2 – Ensemble Learning by Image Type

Our EDA (`file 1-3`) revealed important visual differences based on:

- **Image type:** color vs grayscale
- **Gender**
- **Occlusion level**

#### 🎯 Approach:
We designed a **dual-model ensemble**:
- One model trained on **RGB (color) images**
- One model trained on **grayscale images**

Each model specialized in its input type, improving feature learning and performance.

#### 🏆 Result:
- **Final score: 0.00089** after 30 epochs (file `2-2`)
- Best performance in the competition
- Robust generalization across data types

Further improvement is possible by:
- Increasing underrepresented occlusion samples
- Refining analysis for gender-specific patterns

---

### 3 – MediaPipe Integration

We used **MediaPipe Face Mesh** to extract **468 facial landmarks per face**.

#### 🧩 Notes:
- MediaPipe requires **RGB input** (even if all channels are identical)
- Binary face masks are stored as 3-channel images (though 1 channel would be enough)
- A **stratified train-test split** was created using synthetic labels combining:
  - Database
  - Gender
  - Image color type

> ⚠️ The pipeline is partially complete. Intermediate results are saved in the updated DataFrame for future processing.

---

### 4 – Outlook & Next Steps

Although the model achieved the **top score** in the challenge, further enhancements are possible:

#### 🔄 Gender Balance:
- A **resampling strategy** (as in Data Challenge 1) could help balance male and female samples during EDA
- This could reduce or eliminate remaining **gender bias** in predictions

#### ⚙️ Model Improvements:
- Continue refining **ResNet-18** using:
  - Color/grayscale ensemble learning
  - Class-weighted training for occlusion
  - Potential occlusion-level-specific models

---


## Requirements

The following packages are required to run this project:

pandas==2.2.3
numpy==2.0.2
pillow==11.2.1
opencv-python==4.11.0.86
matplotlib==3.9.4
seaborn==0.13.2
tqdm==4.67.1
scikit-learn==1.6.1
torch==2.7.1
torchvision==0.22.1
mediapipe==0.10.21

css
Copier
Modifier


To install all dependencies, run:

```bash
pip install -r requirements.txt