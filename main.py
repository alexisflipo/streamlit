import streamlit as st 
import pandas as pd
import requests
import json
import plotly.express as px

st.title("Représentation graphique des résultats obtenus par le modèle LDA-TFID")

# Requête en boucle sur l'API
publish_date = []
sentence = []
score=[]
topics_name = []
year=[]
count = 1
for i in range(0,10):
    data = requests.get(f"https://lda-tfidf-api.herokuapp.com/data/data{count}")
    for item in data.json():
        publish_date.append(item['publish_date'])
        sentence.append(item['sentence'])
        score.append(item['score'])
        topics_name.append(item['topic_name'])
        year.append(item['year'])
        i+=1

# Création du DF
df = pd.DataFrame({'publish_date': publish_date, 'sentence':sentence,'score':score, 'topic_name':topics_name, 'year':year })
df_grouped_by_topic = df.groupby(['year','topic_name']).count().reset_index()
hist = px.histogram(df, x="topic_name", color='year',
labels={"year": "Année","topic_name": "Nom du sujet"}, 
title="Répartition des sujets par année").update_layout(yaxis_title="Total de sujets")

line = px.line(df_grouped_by_topic, x="year", y="score", color='topic_name',
labels={
    "year": "Année",
    "score": "Nombre de sujets",
    "topic_name": "Nom du sujet"
},
title="Evolution des sujets par année")

#st.dataframe(df)
st.plotly_chart(hist)
st.plotly_chart(line)


st.text_input("Rentrez le titre d'un article de journal")