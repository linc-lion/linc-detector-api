### Linc Detector Webapp

### Requirements
- Python v3.8+

### Install
To install, clone this repo and cd into the root of the project.

1) Create a virtual environment:

```python3 -m venv venv```

2) Activate the virtual environment. If you're using a Unix-based OS, run:

```source venv/bin/activate```

If you're using Windows, run:

```venv\Scripts\activate```

3) Install the needed python dependencies:

```pip install -r requirements.txt```

### Running Locally

1. Configure Environment Variables

    Add a .env file in the root directory of the project add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as show below will

    ```buildoutcfg
    AWS_ACCESS_KEY_ID = "KEY_ID"
    AWS_SECRET_ACCESS_KEY = "ACCESS_KEY"
    ```
3. Change `app.py` file name to `application.py` Elasticbeanstalk requires the entry point 
flask application file name to be `application.py` but running locally flask looks for
a file named `app.py`.

4. Execute ```flask run``` at the root of the project

You should be able to view the app at 127.0.0.1:5000
