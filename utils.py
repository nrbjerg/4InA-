import torch
from model import Net
import os, shutil

def resetDirectory (folder: str) -> None:
    """ Resets a directory (removes all of the files) """
    # Remove all of the files in the directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def saveModel (model: Net, filename: str):
    """ Saves the current model """
    if (not os.path.exists("models")):
        os.mkdir("models")
        
    filepath = os.path.join(os.getcwd(), "models", filename)
    torch.save(model, filepath)

def loadModel(filename: str) -> Net:
    """ Loads a model from the models directory """
    return torch.load(os.path.join(os.getcwd(), "models", filename))

def loadLatetestModel () -> Net:
    """ Loads the latest model from the models directory """
    files = os.listdir(os.path.join(os.getcwd(), "models"))
    file = str(sorted([int(f.split(".")[0]) for f in files])[-1]) + ".pt"
    model = loadModel(file)
    return model

if (__name__ == "__main__"):
    resetDirectory("./models")
    resetDirectory("./logs")