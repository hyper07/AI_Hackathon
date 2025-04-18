# AI Hackathon at MongoDB Docker Stack

This project is an implementation of a stack based on Docker (docker-compose) using MongoDB, Flask, Jupyter.

## Features

- **MongoDB**: NoSQL database
- **Flask**: Micro web framework for Python
- **Jupyter**: Interactive computing environment

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
  client = MongoClient('mongodb://admin:PassW0rd@apan-mongo:27017/')
  ```


### Jupyter

- **Web Interface**: [http://localhost:8899](http://localhost:8899)

### Flask App

- **Web Interface**: [http://localhost:5010](http://localhost:5010)
- **Web Sample Form**: [http://localhost:5010/form](http://localhost:5010/form)


# HOW TO ACCESS DATABASE AND API
For detailed instructions on how to access and interact with the databases

- **How to access dbs**: [How To Access](https://github.com/hyper07/AI_Hackathon/blob/main/files/work/HowToUse.md).



## Dataset

https://www.kaggle.com/datasets/ibrahimfateen/wound-classification/data

```bash
curl -L -o ./wound-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/ibrahimfateen/wound-classification
```


For more detailed information on each service, please refer to the respective documentation.