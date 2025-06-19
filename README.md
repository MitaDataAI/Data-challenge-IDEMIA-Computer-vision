# Data Challenge 2 with IDEMIA : Computer Vision

# IDEMIA 
IDEMIA is the world leader in identity technologies. It specializes in biometric solutions, secure digital identification, and authentication systems for both governments and private sectors.
![image](https://github.com/user-attachments/assets/30139190-db91-4098-97f9-105a12159ca4)

# By RAKOTONIAINA Pety Ialimita (pety.rakotoniaina@télécom-paris.fr)

# Place during the Data Challenge : Champion ! 

# Structure and Submission Process
The Data Challenge follows the standard principle of a “Kaggle Competition,” based on real-world data and a specific problem. We will be able to download the labeled training data and the test data (without labels, of course) from the Data Challenge website. The predictions we compute using the methods of our choice for the test data must be submitted (in the form of a flat file) on the Data Challenge website. They will be evaluated instantly, placing you on the competition leaderboard. Multiple submissions are, of course, allowed.

# Goal
The primary objective is to predict the percentage of facial occlusion in 30,507 images, based on a training set of 101,341 images. The following image illustrates the key aspects of facial occlusion.
![image](https://github.com/user-attachments/assets/998844b2-c83d-479a-9834-51ec9970d685)!
Red indicates the occluded (hidden) areas of the face, while blue highlights the visible regions. The final image represents a face with 100% visibility.

# Metric
## Objective  
The objective of this challenge is to **regress the percentage of the face that is occluded**.

## Error Calculation
The error is computed as a **weighted mean squared error**:

$$
\text{Err} = \frac{\sum_i w_i (p_i - GT_i)^2}{\sum_i w_i}, \quad w_i = \frac{1}{30} + GT_i
$$

- p_i \: predicted occlusion percentage for sample *i*
-  GT_i \: ground truth occlusion percentage for sample *i*
- w_i \: weight based on occlusion percentage

## Fairness Consideration

To ensure fairness across gender groups, the final score is calculated as:

$$
\text{Score} = \frac{\text{Err}_F + \text{Err}_M}{2} + \left| \text{Err}_F - \text{Err}_M \right|
$$

- \ \text{Err}_F \: weighted error on female samples
- \ \text{Err}_M \: weighted error on male samples

The goal is to **minimize the score**, ensuring both accuracy and fairness across genders.
