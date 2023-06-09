# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Loads and merge messages and categories data from csv files.

    Inputs: 
        messages_filepath: filepath of messages.csv data
        categories_filepath: filepath of categories.csv data
    Output: 
        df (DataFrame): messages and categories merged dataframe
    """
    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataframes
    df = pd.merge(messages, categories, on = 'id')

    return df

def clean_data(df):
    """
        Clean unstructured merged dataframe.
        1. Creates seperate category columns with 0 and 1 flags
        2. Removes duplicates

    Inputs: 
        df: merged dataframe output from load_data 
    Output: 
        df (DataFrame): cleaned dataframe
    """
    # split categories into seperate category values
    categories = df['categories'].str.split(';', expand=True)

    # rename columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # convert category values into 0 and 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    # replace original categories column
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Replace 2 value in related column with 0
    df['related'].replace(2,0, inplace=True)

    return df
    

def save_data(df, database_filename):
    """
        Save dataframe into SQLite database.

    Inputs: 
        df: cleaned dataframe from clean_data
        database_filename: name of the database
    Output: 
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()