# AI Hackathon: Wound Image Classification with MongoDB Vector Search

> **Developed for the VOXEL51 & MongoDB AI Hackathon**

This project demonstrates an end-to-end AI application for medical image classification, leveraging **MongoDB's vector search** to identify wound types from images. The stack is containerized using Docker Compose and integrates **Flask**, **Jupyter**, and **MongoDB** for scalable, reproducible development and deployment.

![AI Hackathon](files/images/AI_Hackathon.png)

![Application](files/images/main.png)

![Training](files/images/training.png)

![Prediction](files/images/prediction.png)

![Setting](files/images/settings.png)

## Project Report (PDF)

You can view the detailed project report or presentation by clicking the link below:

[📄 View Project Report (PDF)](static/files/report.pdf)


## Project Overview

This application was created as part of the **VOXEL51 & MongoDB AI Hackathon**. It enables users to upload wound images and receive automated wound type predictions using a trained deep learning model. The system stores image embeddings in MongoDB and utilizes **vector search** to efficiently retrieve the most similar cases, supporting explainable AI in medical diagnostics.

**Key Features:**
- **Wound Image Classification:** Predicts wound types from uploaded images using a trained neural network.
- **MongoDB Vector Search:** Stores image embeddings and retrieves similar cases for explainable results.
- **Interactive Web UI:** Built with Flask for easy image upload, prediction, and result visualization.
- **Jupyter Integration:** For model training, experimentation, and data exploration.
- **Containerized Stack:** All services run in isolated Docker containers for easy setup and reproducibility.

## Technologies Used

- **MongoDB**: Stores image data and embeddings, provides vector search capabilities.
- **Flask**: Serves the web application and API endpoints.
- **Jupyter**: Supports interactive development and model training.
- **Docker Compose**: Orchestrates multi-container deployment.

## My Contributions

- Designed and implemented the end-to-end pipeline for wound image classification.
- Integrated MongoDB vector search for efficient and explainable image retrieval.
- Developed the Flask web interface for user interaction.
- Automated the environment setup using Docker Compose.

## Building & Running

```sh
# Clone the repository
git clone https://github.com/hyper07/AI_Hackathon.git

# Move to the project directory
cd AI_Hackathon/

# Build and run the containers
docker-compose up -d

# Stop and remove the containers
docker-compose down
```

## Accessing Services

### MongoDB

- **Connection String**:
  ```python
  from pymongo import MongoClient
  client = MongoClient('mongodb://user:pass@hackathon-mongo:27017/')
  ```

### Jupyter

- **Web Interface**: [http://localhost:8899](http://localhost:8899)

### Flask App

- **Web Interface**: [http://localhost:5010](http://localhost:5010)

## Dataset

Wound image dataset from Kaggle:  
https://www.kaggle.com/datasets/ibrahimfateen/wound-classification/data

```bash
curl -L -o ./wound-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/ibrahimfateen/wound-classification
```

## References

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

For more detailed information on each service, please refer to the respective documentation.