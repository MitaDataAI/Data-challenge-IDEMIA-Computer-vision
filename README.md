# Data Challenge 2 with IDEMIA : Computer Vision

# IDEMIA 
IDEMIA is the world leader in identity technologies. It specializes in biometric solutions, secure digital identification, and authentication systems for both governments and private sectors.
![image](https://github.com/user-attachments/assets/30139190-db91-4098-97f9-105a12159ca4)

# By RAKOTONIAINA Pety Ialimita (pety.rakotoniaina@télécom-paris.fr)

# Place during the Data Challenge : Champion ! 

# Structure and Submission Process
The Data Challenge follows the standard principle of a “Kaggle Competition,” based on real-world data and a specific problem. We will be able to download the labeled training data and the test data (without labels, of course) from the Data Challenge website. The predictions we compute using the methods of our choice for the test data must be submitted (in the form of a flat file) on the Data Challenge website. They will be evaluated instantly, placing you on the competition leaderboard. Multiple submissions are, of course, allowed.

# Goal
We have at your disposal 100000 images of human faces, and their occlusion label.
The goal of this challenge is to regress the percentage of the face that is occluded.
We also want to have similar performances on female and male, the gender label is given for the train database.
![image](https://github.com/user-attachments/assets/998844b2-c83d-479a-9834-51ec9970d685)!
Red indicates the occluded (hidden) areas of the face, while blue highlights the visible regions. The final image represents a face with 100% visibility.

# Metrics

The evaluation is based on a **weighted mean squared error (WMSE)**, computed **separately for males and females**, and then combined to produce the final score.

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
