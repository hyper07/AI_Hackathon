# AI Hackathon at MongoDB

## Setup

1. Create and activate the virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the application

```
python -m src.main.app
```

## Running tests

```
pytest src/tests
```

## Dataset

https://www.kaggle.com/datasets/ibrahimfateen/wound-classification/data

```bash
curl -L -o ./wound-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/ibrahimfateen/wound-classification
```

> > > > > > > 74d7d8b (setup)
