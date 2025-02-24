import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """This function receive two csv paths, reads then and merge into a
    new dataset based on id column.

    args:
        messages_filepath -> path of a csv datasource - (String type)
        categories_filepath ->  path outher csv datasource - (String type)

    returns:
        df: Pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, left_on="id", right_on="id")
    return df


def clean_data(df):
    """This function receive a dataframe, clean 'Categories' column spliting by ';'
        and fix with into a new dataframe.

    args:
        df->  Pandas dataframe

    returns:
        df->  Pandas dataframe containing with categories column fixed
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split("-").apply(lambda x: x[0])
    categories.columns = list(category_colnames.values)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-", expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df = df[df["related"] != 2].copy()
    return df


def save_data(df, database_filename):
    """This function save a dataframe into a Table named "clean_table"
     at a SQLite database named "Disaster.db" .

    args:
        df->  Pandas dataframe
        database_filename -> Path where to save the database (String Type)

    returns:
        None

        exemple.:
            save_data(dataframe, "./data")
    """
    engine = create_engine(f"sqlite:///{database_filename}/Disaster.db")
    df.to_sql("clean_data", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
