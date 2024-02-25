# CSE151A-MEW

## Principle Members:

- Albert Chen
- Darren Yu
- Dylan Olivares
- Leo Friedman
- Merrick Qiu
- Micahel Ye
- Nathan Morales
- Yifan Chen

## Abstract

The Glassdoor data set contains 800K+ reviews that employees have given for their jobs. It contains reviews from 1 to 5 for categories such as Career Opportunities, Compensation and Benefits, Company Culture, Management, and Work Life Balance. It also has rankings for whether the employee approves of the CEO, the Company Outlook, and the overall company. Finally it contains textual data the employee has written about the pros and cons of the company as data about location, job title, and date. Using a supervised model on the ratings, rankings, and job titles, and possibly sentiment analysis on the text reviews in the future, we can create a model that predicts the company rating (1-5 stars) of employees based on how they felt about the aspects of the company. This model is useful for employees trying to gauge how important various aspects of a company are or employers trying to improve their company image.

## Preprocessing Explanation

### Data Overview

Our dataset was originally stored in a .csv file, which we converted to .json for ease of use. The dataset has 18 different features and a total of 838,566 observations.

### Missing Data

We explored the ratios of missing data and found that some columns contain different amounts of missing data. Our plan is to drop the columns with a 25% or more ratio of missing data and for the columns with less than 25% missing data we will fill in the missing values with the averages for that specific feature. This means that we will be dropping the 'location' and 'diversity-inclusion' columns because they contain ~35% and ~83% missing values respectively. This cleans up the data and preserves as much of it as possible without skewing it in an unnatural manner.  
The range of the 'work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt', and 'overall_rating' features are already on a convenient whole integer scale from 1 to 5, meaning we do not need to normalize or standardize these feature values. We can easily replace missing values in these columns with their respective averages in order to preserve the overall means. We decided to drop entries with 4 or more missing feature values (~20% missing feature values). This excludes 151,824 entries from the data, bringing the total number of observations down to 686,742.  
Below is a table showing the percentage of missing values for each column.

| Column name         | Percent of Missing Values |
| ------------------- | ------------------------- |
| firm                | 0                         |
| date_review         | 0                         |
| job_title           | 0                         |
| current             | 0                         |
| location            | 35%                       |
| overall_rating      | 0                         |
| work_life_balance   | 18%                       |
| culture_values      | 23%                       |
| diversity_inclusion | 84%                       |
| career_opp          | 18%                       |
| comp_benefits       | 18%                       |
| senior_mgmt         | 19%                       |
| recommend           | 0                         |
| ceo_approv          | 0                         |
| outlook             | 0.2%                      |
| headline            | 0                         |
| pros                | 0                         |
| cons                | 0.00009%                  |

### Encoding

We will also have to encode different non-numerical features. We would do a one-hot encoding for the 'firm' and 'date' columns and an integer encoding for the 'recommend', 'ceo_approv', and 'outlook' columns (these describe positive, mild, negative, and neutral sentiments which can easily be encoded as integers).

### Additions

The 'current' feature describes whether or not the entry is a current or former employee, as well as their duration of employment. We have decided to split this feature into two different columns. One column will describe whether or not the employee is currently or formerly employed as a binary value. The other column will describe the duration of their employment as an integer value starting at 0 (less than one year).

## Reflections

### File Links

[Preprocessing File](/src/preprocess.ipynb)  
[Baseline Model Definition and Training](/src/models/baseline.ipynb)

### Baseline Model Evaluation

Our model’s training mean squared error was approximately 0.03. Our testing mean squared error also came out to be approximately 0.03. The two metrics are nearly the same, indicating that overfitting in our model is very unlikely. Our train and test fraction of variance unexplained (FVU) were roughly equal as well (0.35 and 0.36 respectively) which also indicates that our current model is not overfitting. Gaging our models performance, based on predictions using test data, we see an accuracy of approximately 0.41, precision of 0.29, and recall of 0.41. Our accuracy indicates that the model is on the right track, as it is performing much better than pure guessing, but far from fully optimized. The disparity between precision and suggests that our model tends to report false negatives at higher rate than it does false positives.

| Metric            | Score  |
| ----------------- | ------ |
| Train Error (mse) | 0.0308 |
| Test Error (mse)  | 0.0310 |
| Train FVU         | 0.3544 |
| Test FVU          | 0.3583 |
| Accuracy          | 0.4074 |
| Precision         | 0.2940 |
| Recall            | 0.4074 |

### Where does your model fit in the fitting graph:

Given our test and train errors are very close to each other (Train MSE: 0.0307 vs. Test MSE: 0.0309 and Train FVU: 0.3543 vs. Test FVU: 0.3582), we know that our model is not overfitting. However, since our test and train errors are relatively high and the accuracy of our model is only 0.4073, we know that there is much room for improvement and indicates that our model is underfitting. 

### Next Steps

The next model we’re thinking about is switching to a classification model perhaps with softmax to treat the ratings like discrete values rather than a continuous value since that is more natural in respect to the data’s format.
Currently we’re essentially creating a very large perceptron because we only have one layer, in the future we want to increase the number of layers and units for both the regression model and a potential classification model so more features can be learned within the models.

### Conclusion

Our first model ended with an accuracy of 40% which, although is better than randomly guessing, is not a reliably consistent result and wouldn’t be trustworthy as a method of prediction.  
This may be due to the model architecture as we don’t have any hidden layers limiting our model on how many features it can consult.  
Additionally, our current analysis process is regressive, where we allow the model to output any numerical value, and we use thresholds to assign them to a rating of 1 to 5. The motivation behind this is because we wanted to maintain the comparative aspect that a 2 star rating is worse than 3 stars, but better than 1 star which would only be relevant given continuous values. However, this may not be the analysis that fits most with the data, as the target’s we are trying to reach are discrete.  
Therefore for future improvements, we would consider increasing the number of layers and units in our model, and switching activation functions to transition towards classification.  
We could also consider adding additional text analysis to the headline, pros, and cons as these columns may also contain some more sentiment we can use to augment our decisions.
