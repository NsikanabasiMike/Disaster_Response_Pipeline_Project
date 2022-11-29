import pandas as pd

from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# Load data from sql database and form a dataframe
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

def return_figures():
    """Creates two plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the two plotly visualizations

    """
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    genre_rel_mean = df.groupby('genre')['related'].mean()
    genre_names = list(genre_rel_mean.index)

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_rel_mean
                )
            ],

            'layout': {
                'title': 'Mean of Genres of message based on Related column',
                'yaxis': {
                    'title': "Mean"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    return graphs
def main():
    if __name__ == '__main__':
        main()