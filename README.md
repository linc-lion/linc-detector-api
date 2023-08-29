### Linc Detector Webapp

### Requirements
- Python v3.8+

### Install
To install, clone this repo and cd into the root of the project.

1) Create a virtual environment:

```conda create --name linc-detector-api python=3.8```

2) Activate the virtual environment. If you're using a Unix-based OS, run:

```conda activate linc-detector-api```

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

You should be able to view the REST endpoint at 127.0.0.1:5000/v1/annotate

POSTMAN CURL
```
curl --location --request POST 'http://127.0.0.1:5000/v1/annotate' \
--header 'Authorization: Bearer 1e620008-745c-4e84-be74-81042ab71b1e' \
--form 'file=@"{FILE_PATH_TO_IMAGE}"'
```

note: file path needs the picture name and file extension ie: `/Users/habibamohamed/Downloads/habiba/evaluation/Amboga/PJB_2359.JPG`


### Running Tests

To run the integration tests for the API, follow these steps:

1. Ensure you have dependencies installed before running tests
2. Run the tests using the command `python test_app.py`.

The tests cover various scenarios, including successful image annotation and cases where exceptions are thrown due to missing or invalid data.
