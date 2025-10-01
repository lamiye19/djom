import streamlit as st
from app.tab_chat import chatbot_tab
# from app.tab_nlu import nlu_tab
# from app.tab_chat_nlu import chatbot_nlu_tab

st.set_page_config(page_title="Assistant d'orientation", page_icon="")

chatbot_tab()
# tab1, tab2, tab3 = st.tabs(["Chatbot RAG", "Compréhension (NLU)", "RAG/NLU"])
# with tab1:
#     chatbot_tab()
    #st.markdown("Chat simple")

# with tab2:
#     nlu_tab()
#with tab3:
    #chatbot_nlu_tab()
    #st.markdown("Chat simple")
