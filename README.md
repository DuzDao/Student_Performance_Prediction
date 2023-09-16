#### Table of contents
  1. [Introduction](#intro)
  2. [Data Sample](#sample)
  3. [Results](#res)
  4. [Members](#mem)

Course Project [Course Name: Machine Learning]

# <a name="intro"></a> Project For Course: Machine Learning
To meet the requirements of the course, we used a dataset named "Student performance prediction" (Free Dataset On [Kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics)) and conducted the experiments regarding performance of the ensemble models. </br>

Models we used:
  - Decision Tree Regressor + LightGBM
  - Random Forest Regressor + Gradient Boosting Regressor
  - LightGBM + xgBoost

# <a name="res"></a> Results

- LightGBM + Decision Tree Regressor
  ```python
    python "main.py" --model "lgbm+dtr"
  ```
![image](https://github.com/DuzDao/Student_Performance_Prediction/assets/95222109/f8a86f86-fbd9-4a6c-b33d-92cc705c1f12)

- Gradient Boosting Regressor + Random Forest Regressor
  ```python
    python "main.py" --model "gbr+rfr"
  ```
![image](https://github.com/DuzDao/Student_Performance_Prediction/assets/95222109/1ae22f34-3c96-485f-b1a0-9a82074757f6)

- LightGBM + xgBoost
  ```python
    python "main.py" --model "lgbm+xg"
  ```
![image](https://github.com/DuzDao/Student_Performance_Prediction/assets/95222109/544aa62d-1ee4-4b47-bdca-aeca0b23c9ad)

# <a name="sample"></a> Data Sample
- Dataset Description is available on [Kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics). </br>
- After preprocessing, we have the below data:
  
| gender | parental level of education	| test preparation course |	reading score |	writing score |	math score |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | 4 | 1 | 70 | 78 | 59 | 
| 0 | 0 | 0 | 93 | 87 | 96 |
| 1 | 4 | 0 | 76 | 77 | 57 |

# <a name="mem"></a> Members
My team:
  - Dao Hoang Dung (University ID: 21521972)
  - To Truong Long (University ID: 21521101)
