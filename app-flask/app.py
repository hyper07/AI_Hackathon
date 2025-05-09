import random
import logging
import os
import requests  # Ensure this import is included
import socket  # Add this import

from flask import Flask, render_template, request, jsonify, Response  # Add Response
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
import json  # Add json

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Bootstrap(app)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "./model/classification_model.keras"
DATASET_PATH = "./static/train_dataset"
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://user:pass@hackathon-mongo:27017/")
DEFAULT_DB_NAME = "hackathon"

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

def check_mongo_hostname():
    """
    Checks if the MongoDB hostname can be resolved.
    Returns None if ok, or an error string if not.
    """
    try:
        # Extract hostname from MONGODB_URL
        host = MONGODB_URL.split('@')[-1].split('/')[0].split(':')[0]
        socket.gethostbyname(host)
        return None
    except Exception as e:
        return f"MongoDB hostname '{host}' could not be resolved: {e}"

def get_database():
    """
    Function to get MongoDB database connection.
    Uses consistent connection string and database name.
    """
    # Check DNS resolution before connecting
    dns_error = check_mongo_hostname()
    if dns_error:
        app.logger.error(dns_error)
        return None

    try:
        client = MongoClient(MONGODB_URL)
        # Ping to confirm connection
        client.admin.command('ping')
        dbnames = client.list_database_names()
        app.logger.info(f"MongoDB ('{client.list_database_names()}'")
        # Return the specific database
        return client[DEFAULT_DB_NAME]
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB ('{DEFAULT_DB_NAME}' at '{MONGODB_URL}'): {e}")
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

@app.route('/background', methods=['GET'])
def background():
    return render_template('background.html')


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
        training_progress["total_epochs"] = 20  # Set to the number of epochs train_model_background will run
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

@app.route('/stream-training-logs')
def stream_training_logs():
    def event_stream():
        last_log_index = 0
        last_status = None

        # Send initial status
        status_data = {
            "type": "status",
            "status_text": training_progress.get("status"),
            "epoch": training_progress.get("epoch"),
            "total_epochs": training_progress.get("total_epochs"),
            "message": training_progress.get("message")
        }
        yield f"data: {json.dumps(status_data)}\n\n"
        last_status = json.dumps(status_data, sort_keys=True)

        while True:
            # Send new logs
            logs = training_progress.get("logs", [])
            while last_log_index < len(logs):
                log_entry = logs[last_log_index].rstrip('\n')
                log_data = {"type": "log", "content": log_entry}
                yield f"data: {json.dumps(log_data)}\n\n"
                last_log_index += 1

            # Send status update if changed
            status_data = {
                "type": "status",
                "status_text": training_progress.get("status"),
                "epoch": training_progress.get("epoch"),
                "total_epochs": training_progress.get("total_epochs"),
                "message": training_progress.get("message")
            }
            status_json = json.dumps(status_data, sort_keys=True)
            if status_json != last_status:
                yield f"data: {json.dumps(status_data)}\n\n"
                last_status = status_json

            # If training is done, stopped, or error, send complete and break
            if training_progress.get("status") in ["done", "stopped", "error"]:
                if training_progress.get("status") == "error":
                    error_data = {"type": "error_message", "content": training_progress.get("message", "")}
                    yield f"data: {json.dumps(error_data)}\n\n"
                complete_data = {"type": "complete", "message": training_progress.get("message", "")}
                yield f"data: {json.dumps(complete_data)}\n\n"
                break

            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream")

def train_model_background(total_epochs=20):
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
        return render_template('upload.html', prediction=None, error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', prediction=None, error="No selected file")
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
            return render_template('upload.html', prediction=None, error="Database connection failed. Check logs.")
        
        collection = db['wounded'] 
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "feature_prediction",
                    "queryVector": feature_vector,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "image_path": 1,
                    "class_prediction": 1,
                    "label": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        try:
            results = list(collection.aggregate(pipeline))
            similar_images = [{"image_path": doc["image_path"], "score": doc["score"],"class_prediction": doc["class_prediction"], "label": doc["label"]} for doc in results]
        except Exception as e:
            app.logger.error(f"Error during vector search: {e}")
            return render_template('upload.html', prediction=None, error=f"Vector search failed: {str(e)}. Ensure 'vector_index' is set up correctly on the '{collection.name}' collection for the 'feature_prediction' path.")

        prediction = {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "uploaded_image_path": filepath,
            "similar_images": similar_images
        }
        return render_template('upload.html', prediction=prediction)
    else:
        return render_template('upload.html', prediction=None, error="Invalid file type")

@app.route('/settings', methods=['GET'])
def settings():
    mongo_status = "Unknown"
    db_stats = {"collections": {}}
    model_exists = os.path.exists(MODEL_PATH)
    dns_error = check_mongo_hostname()
    if dns_error:
        mongo_status = f"DNS Error: {dns_error} (Check your MONGODB_URL: '{MONGODB_URL}')"
        return render_template('settings.html', mongo_status=mongo_status, db_stats=db_stats, model_exists=model_exists)
    try:
        db = get_database() # Uses standardized connection and DB name
        if db is not None:
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
            
    return render_template('settings.html', mongo_status=mongo_status, db_stats=db_stats, model_exists=model_exists)

@app.route('/initDB', methods=['GET'])
def init_DB():
    # This function needs a direct client connection to drop/create the database.

    client = None
    dns_error = check_mongo_hostname()
    if dns_error:
        training_progress["logs"].append(f"DNS Error: {dns_error}\n")
        training_progress["status"] = "idle"
        training_progress["message"] = f"DNS Error: {dns_error}"
        return jsonify({"status": "error", "message": dns_error}), 500
    try:
        # Indicate DB init is starting in the logs/status
        training_progress["status"] = "db_init"
        training_progress["message"] = "Initializing database..."
        training_progress["logs"].append("Starting database initialization...\n")

        client = MongoClient(MONGODB_URL)
        client.admin.command('ping') # Check connection
        
        dbnames = client.list_database_names()
        if DEFAULT_DB_NAME in dbnames:
            client.drop_database(DEFAULT_DB_NAME)
            msg = f"Dropped database: {DEFAULT_DB_NAME}"
            app.logger.info(msg)
            training_progress["logs"].append(msg + "\n")
        
        db = client[DEFAULT_DB_NAME]
        collection = db['wounded']
        model = load_model(MODEL_PATH)
        collection = db['wounded']
        counter = 0
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
                                counter += 1
                                msg = f"{counter} image inserted: {img_path}"
                                print(msg)
                                training_progress["logs"].append(msg + "\n")
                        
                        except Exception as e:
                            msg = f"Failed to process {img_path}: {e}"
                            print(msg)
                            training_progress["logs"].append(msg + "\n")
                                
        print(f"Total {counter} images are inserted")

        # Create the vector index for the collection using db.command (Atlas Search)
        try:
            index_command = {
                "createSearchIndexes": "wounded",
                "indexes": [
                    {
                        "name": "vector_index",
                        "definition": {
                            "mappings": {
                                "dynamic": False,
                                "fields": {
                                    "feature_prediction": {
                                        "type": "knnVector",
                                        "dimensions": 2048,
                                        "similarity": "cosine"
                                    }
                                }
                            }
                        }
                    }
                ]
            }
            db.command(index_command)
            msg = "Atlas Search vector index 'vector_index' created on 'feature_prediction' field."
            app.logger.info(msg)
            training_progress["logs"].append(msg + "\n")
        except Exception as e:
            msg = (
                f"Failed to create Atlas Search vector index automatically: {e}\n"
                "You may need to create the index manually in the Atlas UI or via the Atlas API."
            )
            app.logger.warning(msg)
            training_progress["logs"].append(msg + "\n")
        
        app.logger.info("Remember to manually create the 'vector_index' for $vectorSearch if not already present.")
        training_progress["logs"].append("Database initialization complete.\n")
        training_progress["status"] = "idle"
        training_progress["message"] = "Database initialization complete."

        return jsonify({"status": "success", "message": f"Database '{DEFAULT_DB_NAME}' initialized. Collection '{collection.name}' created."})
    except Exception as e:
        app.logger.error(f"Error in init_DB: {e}")
        # Add user-friendly hint for connection refused
        error_message = str(e)
        if "Connection refused" in error_message:
            error_message += (
                "\nHint: Connection was refused. "
                "Please ensure your MongoDB container/service is running and accessible at the configured host/port. "
                f"Current MONGODB_URL: '{MONGODB_URL}'"
            )
        training_progress["logs"].append(f"Error in init_DB: {error_message}\n")
        training_progress["status"] = "idle"
        training_progress["message"] = f"Error in init_DB: {error_message}"
        return jsonify({"status": "error", "message": error_message}), 500
    finally:
        if client:
            client.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
