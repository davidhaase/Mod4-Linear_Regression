# Revamped Kings County Project
---

## Introduction
---
In this readme, we'll review all of the guidelines and specifications for your **revamped** module 1 project. The original mod1 project guidelines can be found [here.](https://github.com/learn-co-curriculum/dsc-v2-mod1-final-project)

## Objectives
----
You will be able to:

- Describe all required aspects of the final project for Module 1 in further detail than a few weeks ago
- Describe what your justifications for the decisions you made regarding feature engineering and feature selection
- Describe all required deliverables and what constitutes a successful project


## The Dataset
---
For this project, you'll be working with the King County House Sales dataset. To help save time, we will be providing the full data set without the missing values you had to deal with during mod1. The following columns **from the original, complete csv** have been transformed for you:

`df['yr_old']=2017 - df['yr_built']`

`df['date'] = pd.to_datetime(df['date'])`

`df['year_sold'] = df['date'].map(lambda x: x.year )`

`df['since_sold'] = 2017  -df['year_sold']`

`df = df[df['price']<4000000]`
`df= df[df['bedrooms']<15]`

`df['price_log'] = np.log(df['price'])`

The dataset can be found in this repo. The file is: "kc_housing_data_for_feat_engineering_lab.csv"

The description of the column names can be found in the column_names.md file in this repository. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions relating to what the data means.

You'll clean, explore, and model this dataset with a multivariate linear regression to predict the sale price of houses as accurately as possible.


## The Deliverables
---
Here are the required deliverables for this project:

A well documented Jupyter Notebook containing any code you've written for this project and comments explaining it.


## Requirements
---

For this project, your Jupyter Notebook should meet the following specifications:

### Revamped Requirements:
- Create **at least** four new features. Two of the features must not be polynomials or interaction terms.
- Explicitly create **at least** one interaction term as a feature.
- Create a base line model with **ALL** of your new features.
- Perform feature selection.
- Explain your thought process in markdown cells.
- Be sure to comment your code thoroughly.
- Run at least three different models. These models should include Lasso, Ridge, or "other" (after feature selection, could be a Ridge/Lasso model with a different alpha).
- After splitting your data into train and test sets, report your final RMSE on test set.

---
### The remaining requirements are the same as the original mod1 Project
---

#### Organization/Code Cleanliness

* The notebook should be well organized, easy to follow, and code should be commented where appropriate.  
    * Level Up: The notebook contains well-formatted, professional looking markdown cells explaining any substantial code.  All functions have docstrings that act as professional-quality documentation
* The notebook is written for technical audiences with a way to both understand your approach and reproduce your results. The target audience for this deliverable is other data scientists looking to validate your findings.

#### Visualizations & EDA

* Your project contains at least 4 meaningful data visualizations, with corresponding interpretations. All visualizations are well labeled with axes labels, a title, and a legend (when appropriate)  
* Your notebook should contain 1 - 2 paragraphs briefly explaining your approach to this project.

#### Model Quality/Approach

* Your model should not include any predictors with p-values greater than .05.  
* Your notebook shows an iterative approach to modeling, and details the parameters and results of the model at each iteration.  
    * **Level Up**: Whenever necessary, you briefly explain the changes made from one iteration to the next, and why you made these choices.  
* You provide at least 1 paragraph explaining your final model.     


## Summary
---
This mini-project is a critical part of the program. It will give you a chance to both bring together all the skills you've learned during the last few weeks into a better version of your initial mod1 project and to practice key "business judgement" and communication skills that you otherwise might not get as much practice with.


### Next Steps:
---
If you are up for it, feel free to revise your original mod1 project with all of the new knowledge you have acquired regarding linear regression. Be sure to comment/explain your decisions and justify why you made them.
