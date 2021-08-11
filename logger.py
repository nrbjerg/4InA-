import os 

def log (file: str, msg: str):
    """ Appends a msg to the log specified """
    with open(os.path.join(os.getcwd(), "logs", file), "a+") as file:
        file.write(msg + "\n")

def logInfo (msg: str):
    log("info.txt", msg)

def logWarning (msg: str):
    log("warning.txt", msg)

def logError (msg: str):
    log("error.txt", msg)