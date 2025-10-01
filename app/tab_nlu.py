import streamlit as st
#from app.utils_nlu import test
import pandas as pd
from app.train import sentences, labels, detect_intention

def nlu_tab():
    st.subheader("Analyse NLU")
    user_q = st.text_input("Posez une question pour analyse NLU")

    if user_q:
        result = detect_intention([user_q])

        st.json({
            "intention": result,
            #"entites": result[0]["entites"]
        })
    
    for q, i in zip(sentences, labels):
        st.write(q)
        result = detect_intention(q)
        st.json({
            "intention": i,
            "detecté": result,
            #"entites": result[0]["entites"]
        })
