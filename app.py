from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import csv

app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

csv.field_size_limit(100000000)
# Load the embedded DataFrame
df_emd = pd.read_csv('embedded_df_10_percent.csv')

# Instantiate ChromaDB PersistentClient and define embedding function
chroma_client_1 = chromadb.PersistentClient(path="my_chromadb")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Get or create the collection
collection_1 = chroma_client_1.get_or_create_collection(name="my_collection_1", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

@app.route('/')
def index():
    message = session.pop('message', None)
    return render_template('index.html', message=message)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    if not query.strip():  # Check if query is empty or only contains whitespace
        session['message'] = 'Please enter a query.'
        return redirect(url_for('index'))

    results = collection_1.query(
        query_texts=[query],
        n_results=10,
        include=['documents', 'distances', 'metadatas']
    )

    documents = results.get('documents', [[]])[0]
    return render_template('results.html', results=documents)

if __name__ == '__main__':
    app.run(debug=True)
