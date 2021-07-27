import streamlit as st
import streamlit as st 
import pandas as pd
import requests
import plotly.express as px
from lda_models import model_result
def app():
    st.title("Project methodology")

    st.header('Preprocessing')

    st.markdown(""" 
    * __Corpus__ : L'ensemble des textes 
    * __Document__ : Element unique au sein d'un corpus 
    * __Tokeniser__ :  transformer les documents en une liste de caracteres (mot d'une phrase par exemple)  
                """) 

    st.subheader('1. Tokenisation')
    st.markdown(""" 
    Premiere étape du projet nécessaire pour appliquer les transformations suivantes
    """)

    st.subheader('2.a Stemming ')
    st.markdown(""" 
    Permet de réduire les mots et de les standardiser en enlevant les __préfixes__ et les __suffixes__
    """)

    st.subheader('2.b Lemmatisation ')
    st.markdown(""" 
    __ Méthode utilisée sur ce projet:__ Permet de standardiser les mots en ne gardant que la __racine__
    """)

    st.subheader('3. StopWords ')
    st.markdown(""" 
    Module nltk permettant d'obtenir une liste de mots sans valeur informative puis de supprimer ceux-ci 
    """) 
    st.code('from nltk.corpus import stopwords')

    st.subheader('4. Preprocessing manuel ')
    st.markdown(""" 
    * Suppression des mots de moins de 3 caracteres
    * Mise en minuscule de tout le corpus 
    * Suppression de la ponctuation 

    """) 

    st.header('Model')

    st.subheader('1. TF-IDF ')

    st.markdown(""" 
    Le TF-IDF est une méthode de pondération souvent utilisée en recherche d'information et en particulier dans la fouille de textes.
    Cette mesure statistique permet d'évaluer l'importance d'un terme contenu dans un document, relativement à une collection ou un corpus.

    Le poids augmente proportionnellement au nombre d'occurrences du mot dans le document. La méthode donne plus de poids aux termes les moins fréquents du corpus considérés 
    comme plus discriminants.
    """) 

    st.image('./images/TFIDF.png')

    st.subheader('2. Latent Dirichlet Allocation ')

    st.markdown(""" 
    Modele probabiliste permettant de clusturiser un corpus de documents. 
    """) 

    st.image('./images/lda.png')

    # st.text_area("Merci d'entrer une phrase en anglais : ")

    st.title("Représentation graphique des résultats obtenus par le modèle LDA-TFID")

    # Requête en boucle sur l'API
    publish_date = []
    sentence = []
    score = []    
    topics_name = []
    year = []
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
    