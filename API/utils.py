import pickle

def load_models(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)







