import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load('../models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Graph 2 - Categories Distribution
    remove_col = ['id', 'message', 'original', 'genre']
    category_df = df.loc[:, ~df.columns.isin(remove_col)]
    category_names = list(category_df.columns)
    category_perc = 100*(category_df.sum(axis=0)/category_df.shape[0])

    # Graph 3 - Top 10 Categories Distribution with Genre

    top_10_cat = list(category_df.sum(axis=0).sort_values(ascending=False).head(10).index)

    def category_genre(df, genre):
        genre_df = df.loc[df['genre'] == genre]
        genre_df_cat = genre_df[top_10_cat]
        genre_count = genre_df_cat.sum(axis=0)
        return genre_count
    
    category_direct = category_genre(df, 'direct')
    category_news = category_genre(df, 'news')
    category_social = category_genre(df, 'social')
    
    
    # create visuals
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
                    x=category_names,
                    y=category_perc
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30
                },
                'yaxis': {
                    'title': "Percent (%)"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_direct,
                    y=top_10_cat,
                    name='Direct',
                    orientation='h'
                ),
                Bar(
                    x=category_news,
                    y=top_10_cat,
                    name='News',
                    orientation='h'
                ),
                Bar(
                    x=category_social,
                    y=top_10_cat,
                    name='Social',
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories with Genre',
                'xaxis': {
                    'title': "Percent (%)"
                },
                'yaxis': {
                    'title': "Category"
                },
                'barmode': 'stack'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()