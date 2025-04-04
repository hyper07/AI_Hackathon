from pymongo import MongoClient

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


if __name__ == "__main__":
    main() 