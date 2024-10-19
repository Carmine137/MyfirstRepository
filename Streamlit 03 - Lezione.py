# Intro
import streamlit as st
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()


# Cache Memory
# In some cases, the code can take a long time to run. To avoid wasting time each time the page is updated,
# we can use the @st.cache_data decorator. 
# It allows to store a value in memory such that if we refresh the Streamlit page with a "Re-run" we always get the same result.

# (a) Create a Python .py script on VSCode or Spyder.
#(b) Copy the following command lines into your Python script and save the file.


@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20) 
st.write(a) 
st.write(b)


# (c) Using the Terminal commands, display the associated Streamlit web page and click on "Re-run" several times. 
# The values of a and b, which are supposed to be random, always remain the same.
import pickle
model_path = r"DS-WorldTemperature Proj\Coding\Part 5 - Streamlit\model"
loaded_model = pickle.load(open(model_path, 'rb'))