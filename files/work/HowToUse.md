```python
import psycopg, os
from PIL import Image
from IPython.display import display
import pandas as pd
from tabulate import tabulate
from pymongo import MongoClient
```

---

# FOR MongoDB


```python
from pymongo import MongoClient
client = MongoClient('mongodb://admin:PassW0rd@apan-mongo:27017/')
```


```python
dbnames = client.list_database_names()
dbnames
```




    ['admin', 'config', 'local']




```python
db = client.test
```


```python
collection = db.test
```


```python
collection
```




    Collection(Database(MongoClient(host=['apan-mongo:27017'], document_class=dict, tz_aware=False, connect=True), 'test'), 'test')




```python
#collection = db.test

import datetime 
from datetime import datetime

post = {'singer': 'Louis Armstrong',
        "song": "What a wonderful world",
        "tags":["jazz", "blues"],
        "date": datetime.now()
}

post_id = collection.insert_one(post).inserted_id
```


```python
print('Our first post id: {0}'.format(post_id))
print('Our first post: {0}'.format(post))
```

    Our first post id: 67cc5ab298c392a2f6b3d821
    Our first post: {'singer': 'Louis Armstrong', 'song': 'What a wonderful world', 'tags': ['jazz', 'blues'], 'date': datetime.datetime(2025, 3, 8, 14, 56, 50, 401266), '_id': ObjectId('67cc5ab298c392a2f6b3d821')}



```python
collection.drop()
```


```python
client.close()
```


```python

```


```python

```


```python

```

---
# APIs

### Call API for flask app.


```python
import urllib3
import json

http = urllib3.PoolManager()

try:
    response = http.request('GET', 'http://apan-flask-app:5010/api/test')

    if response.status == 200:
        data = json.loads(response.data.decode('utf-8'))
        print(f"New API data: {data}")
    else:
        print(f"An error occurred: {response.status}")
except urllib3.exceptions.HTTPError as e:
    print(f"An error occurred: {e}")
```

    New API data: {'random_number': 8}



```python

```


```python

```


```python

```
