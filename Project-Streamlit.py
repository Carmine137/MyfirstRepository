
#Intro

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()

#Creating Title
pages=["Introduction", "Data Exploration", "DataVizualization", "Modelling", "Conclusion"]
page=st.sidebar.radio("Go to", pages)

st.sidebar.title("Summary")


if page == pages[0] :
    st.title("DataScientest - Wolrd Temperature Project - AUG24")
    st.write("Test per capire se esce scritto qualcosa")





# Parte da eliminare
"""Aggiungere descrizione:
    Aggiungere nome dei partecipanti
    aggiungere nomi delle pagine
    """