# Udacity Starbucks Capstone Project
## Table of Contents
1. [Project Overview](#project-overview)
2. [Questions](#questions)
3. [Installation](#installation)
6. [File Descriptions](#file-descriptions)
7. [Results](#results)
8. [Acknowledgements](#acknowledgements)

## Project Overview
The Starbucks Capstone Project is part of the Udacity Data Science Nanodegree program. The objective of this project is to analyse simulated customer behavior data provided by Starbucks to understand the impact of various promotional offers on customer purchases, and create a recommendation system for personalised offers.

The project consists of three main parts:

1. Data Preprocessing: Cleaning, merging, and transforming the provided datasets to create a single comprehensive dataset for analysis.
2. Exploratory Data Analysis: Exploring the dataset to gain insights into customer behavior and identify patterns or trends.
3. Machine Learning: Building a machine learning model to predict a customer's likelihood of completing an offer after viewing it which could feed into creating a recommendation system based for which offers to send to which customers.

## Questions
We'll be answering the following four main questions based on the Starbucks data:

1. Does gender correlate to how much someone spends at Starbucks?
2. How many customers viewed then completed offers? And how many customers completed offers without seeing them first?
3. Which offer type is the most likely to be completed after viewed?
4. What are the attributes that most contribute to a customer's likelihood of completing an offer after viewing it?

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.

Alternatively, if you do not wish to use Anaconda, you will need to install Jupyter Notebook and the required Python libraries.

```python
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

The code should run with no issues using Python versions 3.*.

## File Descriptions
* data/portfolio.json - JSON file containing information about the different promotional offers.
* data/profile.json - JSON file containing demographic data about each customer.
* data/transcript.json - JSON file recording events and transactions related to the offers.
* Starbucks_Capstone_notebook.ipynb: - Jupyter Notebook to showcase work related to the above questions. The notebook contains markdown cells to assist in walking through the thought process for individual steps, which starts with data cleaning before looking at each question individually.

## Results
The main findings of the code can be found at the post available [here](https://medium.com/@sarinapatel1213/how-can-we-boost-us-college-completion-rates-6f29efd3dd15). 

## Acknowledgements
The dataset used in this project was provided by Starbucks as part of the Udacity Data Science Nanodegree program. Special thanks to Starbucks for making the dataset available for educational purposes.
