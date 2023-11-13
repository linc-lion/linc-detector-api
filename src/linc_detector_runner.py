import bentoml
from sklearn.base import BaseEstimator

if __name__ == "__main__":

    bento_model = bentoml.pytorch.load_model("habiba:latest")

    print(bento_model)
    print(bento_model.tag)
    print(bento_model.path)
    print(bento_model.custom_objects)
    print(bento_model.info.metadata)
    print(bento_model.info.labels)

    my_runner: bentoml.Runner = bento_model.to_runner();
