import streamlit as st
import streamlit as st
import streamlit as st 
import pandas as pd
import plotly.express as px
from lda_models import model_result
def app():
    st.title('Model test on unknown variable: ')
    unknown_text=st.text_input("Rentrez le titre d'un article de journal")

    topic, result=model_result(unknown_text)

    st.markdown(f"__Topic Name :__ {topic.capitalize()}")
    st.subheader(result)