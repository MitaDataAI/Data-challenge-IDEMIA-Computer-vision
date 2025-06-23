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