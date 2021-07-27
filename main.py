import streamlit as st 
import pandas as pd
import requests
import json
import plotly.express as px
import app1
import app2
from lda_models import model_result
PAGES = {
    "Préprocessing & Analyse": app1,
    "Test du modèle": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()



