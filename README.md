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

## Preprocessing Explanation

Our dataset was originally stored in a .csv file, which we converted to .json for ease of use. The dataset has 18 different features and a total of 838,566 observations. We explored the ratios of missing data and found that some columns contain different amounts of missing data. Our plan is to drop the columns with a 25% or more ratio of missing data and for the columns with less than 25% missing data we will fill in the missing values with the averages for that specific feature. This means that we will be dropping the 'location' and 'diversity-inclusion' columns because they contain ~35% and ~83% missing values respectively. This cleans up the data and preserves as much of it as possible without skewing it in an unnatural manner. The range of the 'work_life_balance', 'culture_values', 'career_opp', 'comp_benefits', 'senior_mgmt', and 'overall_rating' features are already on a convenient whole integer scale from 1 to 5, meaning we do not need to normalize or standardize these feature values. We can easily replace missing values in these columns with their respective averages in order to preserve the overall means. We decided to drop entries with 4 or more missing feature values (~20% missing feature values). This excludes 151,824 entries from the data, bringing the total number of observations down to 686,742. We will also have to encode different non-numerical features. We would do a one-hot encoding for the 'firm' and 'date' columns and an integer encoding for the 'recommend', 'ceo_approv', and 'outlook' columns (these describe positive, mild, negative, and neutral sentiments which can easily be encoded as integers). The 'current' feature describes whether or not the entry is a current or former employee, as well as their duration of employment. We have decided to split this feature into two different columns. One column will describe whether or not the employee is currently or formerly employed as a binary value. The other column wil describe the duration of their employment as an integer value starting at 0 (less than one year). 

(column name, percent of missing values)
('firm', 0.0) 
('date_review', 0.0) 
('job_title', 0.0) 
('current', 0.0) 
('location', 0.35457912674732817) 
('overall_rating', 0.0) 
('work_life_balance', 0.17875039054767305) 
('culture_values', 0.22821459491560592) 
('diversity_inclusion', 0.8377396650949359) 
('career_opp', 0.17589670938244575) 
('comp_benefits', 0.17897458279968423) 
('senior_mgmt', 0.18588399720475193) 
('recommend', 0.0) 
('ceo_approv', 0.0) 
('outlook', 0.0) 
('headline', 0.002650954128834224) 
('pros', 0.0) 
('cons', 9.540095830262615e-06)

(0, 108627) (1, 358636) (2, 201886) (3, 17593) (4, 3196) (5, 3577) (6, 64706) (7, 80057) (8, 288) (9, 0) (10, 0) (11, 0) (12, 0) (13, 0) (14, 0) (15, 0) (16, 0) (17, 0) (18, 0)

151,824
686,742