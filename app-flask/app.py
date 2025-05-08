import random
import logging
import os
import requests  # Ensure this import is included

from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from IPython.display import display, Image as IPImage
from PIL import Image
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
from threading import Thread
import time
from tensorflow.keras.callbacks import Callback
import io
import sys

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Bootstrap(app)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "./model/classification_model.keras"
DATASET_PATH = "./train_dataset"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Global variable to store training progress
training_progress = {
    "status": "idle",  # idle, running, done
    "epoch": 0,
    "total_epochs": 0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "message": "",
    "logs": []  # Add this line
}

# Add a global stop flag
stop_training_flag = {"stop": False}

class LogCapture(io.StringIO):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def write(self, s):
        # Split on both \n and \r to capture progress bar updates and logs
        for line in s.replace('\r', '\n').split('\n'):
            if line.strip():
                self.log_list.append(line + '\n')
        super().write(s)

def custom_generator(generator):
    for batch_x, batch_y in generator:
        dummy_labels = np.zeros((batch_y.shape[0], 2048))
        yield batch_x, {"class_output": batch_y, "feature_output": dummy_labels}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_database():
    """
    Function to get MongoDB database connection.
    Uses consistent connection string and database name.
    """
    # Ensure this connection string and DB name are correct for your setup.
    CONNECTION_STRING = "mongodb://user:pass@hackathon-mongo:27017/"
    DB_NAME = "hackathon"
    try:
        client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        # Ping to confirm connection
        client.admin.command('ping')
        # Return the specific database
        return client[DB_NAME]
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB ('{DB_NAME}' at '{CONNECTION_STRING}'): {e}")
        return None

def get_predictions(model, test_image_path):
    
    # Load and preprocess the test image
    img_height, img_width = 256, 256  # Ensure this matches the model's input size
    test_image = load_img(test_image_path, target_size=(img_height, img_width))  # Resize image
    test_image_array = img_to_array(test_image)  # Convert to numpy array
    test_image_array = test_image_array / 255.0  # Normalize pixel values to [0, 1]
    test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(test_image_array)
    class_predictions = predictions[0]  # Class predictions
    feature_predictions = predictions[1]  # 1536-dimensional array
    return {"class": class_predictions[0], "feature":feature_predictions[0]}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Example: process form data
        form_data = request.form.to_dict()
        # You can add logic to handle the form data here
        return render_template('form.html', form_data=form_data, message="Form submitted successfully!")
    return render_template('form.html')

@app.route('/train/stop', methods=['POST'])
def stop_training():
    stop_training_flag["stop"] = True
    training_progress["message"] = "Stopping training..."
    return jsonify({"status": "stopping"})

@app.route('/train', methods=['GET', 'POST'], endpoint='train_model')
def train_model():
    class_counts = {}
    if os.path.exists(DATASET_PATH):
        for class_name in os.listdir(DATASET_PATH):
            class_dir = os.path.join(DATASET_PATH, class_name)
            if os.path.isdir(class_dir):
                file_count = len([
                    f for f in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, f))
                ])
                class_counts[class_name] = file_count

    if request.method == 'POST':
        if training_progress["status"] == "running":
            return jsonify({"status": "error", "message": "Training already running."}), 400
        training_progress["status"] = "running"
        training_progress["epoch"] = 0
        training_progress["train_acc"] = 0.0
        training_progress["val_acc"] = 0.0
        training_progress["message"] = "Training started."
        stop_training_flag["stop"] = False  # Reset stop flag
        thread = Thread(target=train_model_background)
        thread.start()
        return jsonify({"status": "started", "message": "Training started."})

    return render_template('train_overview.html', class_counts=class_counts)

@app.route('/train_status', methods=['GET'])
def train_status():
    # Return last 50 log lines for brevity
    logs = training_progress.get("logs", [])
    return jsonify({**training_progress, "logs": logs[-50:]})

def train_model_background(total_epochs=1):
    global training_progress
    img_height, img_width = 256, 256
    batch_size = 32

    # Clear logs at start
    training_progress["logs"] = []

    # Redirect stdout/stderr to log buffer
    log_capture = LogCapture(training_progress["logs"])
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = log_capture, log_capture

    try:
        datagen = ImageDataGenerator(
            rescale=1.0/255,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.1
        )
        original_train_generator = datagen.flow_from_directory(
            DATASET_PATH,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )
        original_val_generator = datagen.flow_from_directory(
            DATASET_PATH,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=True
        )

        train_generator = custom_generator(original_train_generator)
        val_generator = custom_generator(original_val_generator)

        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(img_height, img_width, 3)
        )
        base_model.trainable = False

        inputs = Input(shape=(img_height, img_width, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        feature_output = Dense(2048, activation="relu", name="feature_output")(x)
        class_output = Dense(original_train_generator.num_classes, activation="softmax", name="class_output")(feature_output)
        model = Model(inputs=inputs, outputs=[class_output, feature_output])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={"class_output": "categorical_crossentropy", "feature_output": "mean_squared_error"},
            metrics={"class_output": "accuracy"}
        )

        steps_per_epoch = original_train_generator.samples // batch_size
        validation_steps = original_val_generator.samples // batch_size

        # Custom training loop to check stop flag
        for epoch in range(total_epochs):
            if stop_training_flag["stop"]:
                training_progress["status"] = "stopped"
                training_progress["message"] = "Training stopped by user."
                break
            history = model.fit(
                train_generator,
                epochs=epoch+1,
                initial_epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=[TrainingProgressCallback()]
            )
        else:
            model.save(MODEL_PATH)
            training_progress["status"] = "done"
            training_progress["train_acc"] = float(history.history["class_output_accuracy"][-1])
            training_progress["val_acc"] = float(history.history["val_class_output_accuracy"][-1])
            training_progress["message"] = "Training completed."


        db = get_database()
        if db is None:
            training_progress["status"] = "error"
            training_progress["message"] = "Database connection failed during post-training data insertion. Check logs."
            # Restore stdout/stderr before returning from the thread
            sys.stdout, sys.stderr = old_stdout, old_stderr
            return # Exit the thread
        
        # Ensure 'wounded' collection is intended for vector search and has 'vector_index'.
        collection = db['wounded'] 
        # Loop through labeled folders
        for label in os.listdir(DATASET_PATH):
            label_dir = os.path.join(DATASET_PATH, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            with Image.open(img_path).convert("RGB") as img:
                                width, height = img.size
                                predictions = get_predictions(model, img_path)

                                # Convert numpy arrays to lists for JSON serialization
                                class_prediction = predictions['class'].tolist() if isinstance(predictions['class'], np.ndarray) else predictions['class']
                                feature_prediction = predictions['feature'].tolist() if isinstance(predictions['feature'], np.ndarray) else predictions['feature']

                                doc = {
                                    "image_path": img_path,
                                    "label": label,
                                    "width": width,
                                    "height": height,
                                    "class_prediction": class_prediction,
                                    "feature_prediction": feature_prediction
                                }

                                collection.insert_one(doc)
                                print(f"Inserted: {img_path}")
                        except Exception as e:
                            print(f"Failed to process {img_path}: {e}")



    except Exception as e:
        training_progress["status"] = "error"
        training_progress["message"] = f"Error: {str(e)}"
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        training_progress["epoch"] = epoch + 1
        training_progress["train_acc"] = float(logs.get("class_output_accuracy", 0.0))
        training_progress["val_acc"] = float(logs.get("val_class_output_accuracy", 0.0))
        msg = f"Epoch {epoch+1} completed."
        training_progress["message"] = msg
        training_progress["logs"].append(msg)

@app.route('/upload', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'GET':
        return render_template('upload.html')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        model = load_model(MODEL_PATH)
        img_height, img_width = 256, 256
        test_image = load_img(filepath, target_size=(img_height, img_width))
        test_image_array = img_to_array(test_image) / 255.0
        test_image_array = np.expand_dims(test_image_array, axis=0)
        predictions = model.predict(test_image_array)
        class_predictions = predictions[0]
        feature_predictions = predictions[1]

        predicted_class = int(np.argmax(class_predictions, axis=1)[0])
        predicted_label = str(predicted_class)

        feature_vector = feature_predictions[0].tolist() if isinstance(feature_predictions[0], np.ndarray) else feature_predictions[0]
        
        db = get_database()
        if db is None:
            return jsonify({"error": "Database connection failed. Check logs."}), 500
        
        # Ensure 'wounded' collection is intended for vector search and has 'vector_index'.
        # The field 'feature_prediction' must exist in documents within this collection.
        collection = db['wounded'] 
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", # This index must exist on the 'wounded' collection.
                    "path": "feature_prediction", # Field containing vectors in 'wounded' documents.
                    "queryVector": feature_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "image_path": 1, # Ensure 'wounded' documents have 'image_path'.
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        try:
            results = list(collection.aggregate(pipeline))
            similar_images = [{"image_path": doc["image_path"], "score": doc["score"]} for doc in results]
        except Exception as e:
            app.logger.error(f"Error during vector search: {e}")
            # This error might indicate the 'vector_index' is missing or not correctly configured.
            return jsonify({"error": f"Vector search failed: {str(e)}. Ensure 'vector_index' is set up correctly on the '{collection.name}' collection for the 'feature_prediction' path."}), 500

        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "similar_images": similar_images
        })
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/settings', methods=['GET'])
def settings():
    mongo_status = "Unknown"
    db_stats = {"collections": {}}
    try:
        db = get_database() # Uses standardized connection and DB name
        if db:
            mongo_status = f"Connected. Database '{db.name}' is accessible."
            collection_names = db.list_collection_names()
            if not collection_names:
                mongo_status += " (No collections found in this database)"
            
            for name in collection_names:
                db_stats["collections"][name] = db[name].count_documents({})
        else:
            # get_database() returned None, error already logged by it.
            mongo_status = "Error: Failed to connect to MongoDB or the specified database. Check application logs for details."
            # db_stats remains empty

    except Exception as e:
        # Catch any other errors during stats collection
        mongo_status = f"Error processing database stats: {str(e)}"
        app.logger.error(f"Error in settings route while fetching stats: {e}")
            
    return render_template('settings.html', mongo_status=mongo_status, db_stats=db_stats)

@app.route('/initDB', methods=['GET'])
def init_DB():
    # This function needs a direct client connection to drop/create the database.
    CONNECTION_STRING = "mongodb://user:pass@hackathon-mongo:27017/" 
    DB_NAME = "hackathon" 
    client = None
    try:
        client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Check connection
        
        dbnames = client.list_database_names()
        if DB_NAME in dbnames:
            client.drop_database(DB_NAME)
            app.logger.info(f"Dropped database: {DB_NAME}")
        
        db = client[DB_NAME]
        # Create 'wounded' collection (or any other essential collections)
        wounded_collection = db['wounded']
        # Insert a sample document to ensure collection creation and provide info
        wounded_collection.insert_one({
            "init": True, 
            "message": "This is the 'wounded' collection. Ensure 'feature_prediction' field and 'vector_index' are set up if used for search.",
            "image_path": "example/initial_image.jpg", # Example field
            "feature_prediction": [0.0] * 2048 # Example placeholder for a feature vector
        })
        app.logger.info(f"Initialized database '{DB_NAME}' and collection '{wounded_collection.name}'.")
        
        # IMPORTANT: Vector index creation (e.g., for $vectorSearch)
        # The index named "vector_index" on the "feature_prediction" path
        # needs to be created on your MongoDB deployment (e.g., via Atlas UI, Atlas API, or specific admin commands).
        # Pymongo's `create_index` is typically not used for these advanced search indexes.
        # Example: db.adminCommand({
        #   "createSearchIndexes": "wounded",
        #   "indexes": [{ "name": "vector_index", "definition": { "mappings": { "dynamic": True, "fields": { "feature_prediction": { "type": "knnVector", "dimensions": 2048, "similarity": "cosine"}}}}}]
        # }) - This is an Atlas Search specific command. Syntax varies.
        app.logger.info("Remember to manually create the 'vector_index' for $vectorSearch if not already present.")

        return jsonify({"status": "success", "message": f"Database '{DB_NAME}' initialized. Collection '{wounded_collection.name}' created."})
    except Exception as e:
        app.logger.error(f"Error in init_DB: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if client:
            client.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
