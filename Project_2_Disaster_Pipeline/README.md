# Disaster Response Pipeline Project
## Table of Contents:
1. [Project Description](#project-description)
2. [Getting Started](#getting-started)
3. [Project Structure](#project-structure)
4. [Acknowledgements](#acknowledgements)

## Project Description:
This project is part of the Data Science Nanodegree Program by Udacity. The purpose of the project is to build a Natural Language Processing (NLP) model that categorizes messages sent during disasters into different categories in order to streamline disaster response efforts. The project includes a Flask web app that enables an emergency worker to input a new message and get classification results in several categories.

## Getting Started:
### Prerequisites
* Python 3.6+
* Libraries: pandas, numpy, sqlalchemy, flask, nltk, scikit-learn, joblib, plotly
### Instructions
1. Run the following commands in the project's root directory to set up the database and model:
    * To run ETL pipeline that cleans data and stores the table in a database:
    ```python
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
    * To run ML pipeline that trains classifier model and saves it:
    ```python
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
2. Run the ML pipeline that trains the classifier model and saves it as a pickle file:
    ```python
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```
3. Navigate to the app directory: 
    ```python
    cd app
    ```
4. Run the web app:
    ```python
    python run.py
    ```
5. Go to [http://127.0.0.1:300](http://127.0.0.1:3000/) to view the web app.

![Disaster Response App](https://github.com/s-patel12/data_science_nanodegree/blob/main/Project_2_Disaster_Pipeline/web_app_screenshot.png)
NOTE: If classifier.pkl and DisasterResponse.db already exist, only complete step 3 onwards.

## Project Structure:
```bash
data/
|- disaster_categories.csv  # categories data
|- disaster_messages.csv    # messages data
|- DisasterResponse.db      # database to save clean data
|- process_data.py          # ETL pipeline script

models/
|- train_classifier.py      # ML pipeline script
|- classifier.pkl           # saved model

app/
|- run.py                   # Flask file that runs the web app
|- templates/
|  |- master.html           # main page of the web app
|  |- go.html               # classification result page of the web app

README.md                    # project description
```

## Acknowledgements:
* [Figure Eight](https://www.figure-eight.com/) for providing the messages and categories dataset.
