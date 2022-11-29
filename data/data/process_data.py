import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

def load_data(messages_filepath, categories_filepath):
    
    """
    Load disaster response data for cleaning.

    Merges messages and categories datasets together to form a dataframe

    Parameters:
            dataset (str): file paths for messages and categories data

    Returns:
          df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='outer', on='id')
    return df

def clean_data(df):
    
    """
    Load df for cleaning.

    Splits the values in the categories column on ; character and turns 
    each value to a separate column
    
    Slices each value up to the second to last character of each string
    Uses the first row of categories dataframe to create column names for
    the categories data
    
    Renames columns of categories with new column names.
    Gets the last character of each value from each column and converts 
    to integer
    
    Drops the earlier categories column from df
    Concats categories dataframe to df
    Drops rows with multiclass category
    
    Parameters:
           Dataframe(df)
    Returns:
          df
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    #Slice up to the second to last character of each string    
    category_colnames = [val[:-2] for val in categories.loc[0,:].values.tolist()]
    # Rename categories column names
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.replace('[^\d]', '')

        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # Drop rows where category value is other than 0 and 1
    df.drop(np.where(categories.loc[:, :] == 2)[0],inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    """
    Loads df to save to a sql database
    
    Saves df to message_categories table in DisasterResponse database
    
    Parameters: 
           dataframe(df), database file name
    Returns: 
         None
    """
    
    engine = create_engine(database_filename)

    df.to_sql('message_categories', engine, if_exists='replace', index=False)    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data('disaster_messages.csv', 'disaster_categories.csv')

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, 'sqlite:///DisasterResponse.db')
        
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

engine = create_engine('sqlite:///DisasterResponse.db')
df1 = pd.read_sql('message_categories', con = engine)
print(df1.head())