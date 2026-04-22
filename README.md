# flood-ale
Predicts flood (yes/no) using supervised, batch and model-based learning methods

**Algorithms used:**
---
Classification algorithms for supervised learning like KNN, Decision Tree, Random Forest, SVM, XGBoost, and Logistic Regression were used for this problem

**Dataset:**
---
The dataset used for this model is ‘65 years of weather data of Bangladesh,' collected from the Bangladesh Meteorological Department (BMD) that was preprocessed and uploaded on Kaggle. In addition, data from other sources like annual reports, newspapers, credible sites, etc. were aggregated in this dataset. 
80% of the data of this dataset was used for training and 20% for validation.

<details>
<summary>Citation</summary>
```
@inproceedings{gauhar2021prediction,
    title={Prediction of Flood in Bangladesh using k-Nearest Neighbors Algorithm},
    author={Gauhar, Noushin and Das, Sunanda and Moury, Khadiza Sarwar},
    booktitle={2021 2nd International Conference on Robotics, Electrical and Signal Processing Techniques (ICREST)},
    pages={357--361},
    year={2021},
    organization={IEEE}
}
```
</details>


**Pipeline used:**
---
1. Relevant libraries were imported, and the dataset was loaded, previewed and skimmed.
2. Missing values were handled and data preprocessed. 
3. Dataset visualized before and after data preprocessing, dataset split, and feature scaled
4. Base models of algorithms were created
5. Base model analyzed using Confusion Matrices, Classification Report, and Cross-Validation
6. Hyperparameter tuning of models was done using GridSearchCV.
7. Best model evaluated using Confusion Matrices, Classification Report, and F1 score (harmonic mean of precision and recall)

**Result:**
---
Based on all evaluation criteria, XGBoost performed best for flood prediction, after that Random forest and Decision tree performed well accordingly. To predict floods from test data, few data instances were manually collected from the ‘Climate Information System’ of BARC, Bangladesh. These test data perfectly predicted flood using the tuned XGBoost model.




