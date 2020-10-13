# -*- coding: utf-8 -*-
import nltk
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("python-Levenshtein")
from dialogue_agent import Dialogue_Agent
nltk.download('punkt')
nltk.download('stopwords')
da = Dialogue_Agent("dialog_acts.dat","restaurant_info.csv")
da.start_dialogue()
