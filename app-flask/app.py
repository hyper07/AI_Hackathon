from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from pymongo import MongoClient
import random
import logging
import os
import requests  # Ensure this import is included

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Bootstrap(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def get_database():
    """
    Function to get MongoDB database connection
    """
    # Connection string - replace with your actual MongoDB connection string if needed
    CONNECTION_STRING = "mongodb://localhost:27017"
    
    # Create a connection to MongoDB
    client = MongoClient(CONNECTION_STRING)
    
    # Return the database (creates it if it doesn't exist)
    return client['mongodb_hackathon']

def main():
    """Main entry point for the application."""
    # Get the database
    db = get_database()
    
    # Example of accessing a collection
    collection = db["sample_collection"]
    
    # Example of inserting a document
    # collection.insert_one({"name": "Example", "value": 42})
    
    # Example of finding documents
    # documents = collection.find({})
    # for doc in documents:
    #     print(doc)
    
    print("MongoDB connected successfully!")
    return 'MongoDB connected successfully!'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route('/api/test', methods=['GET'])
def test():
    
    return main()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
