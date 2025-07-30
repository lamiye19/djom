import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import datetime
    
def generate_txt(chat_history):
    lines = []
    for message in chat_history:
        role = message.type
        content = message.content
        lines.append(f"{role} : {content}")
    return "\n\n".join(lines)
    
def get_context(query):
    docs = retriever.invoke(query)[:3]
    return "\n".join(doc.page_content for doc in docs)

def get_chain(model_name):
    llm = ChatGroq(
        model_name=model_name,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Tu es Djom, un conseiller d’orientation scolaire et professionnelle.
    Tu aides les jeunes à choisir leur parcours d'études ou leur métier selon leurs passions, 
        leurs compétences, et les réalités locales au Togo. 
        Donne des conseils clairs en t’appuyant sur leurs questions et les éléments déjà évoqués dans la conversation.

    Contexte extrait de documents utiles :
    {context}

    Historique de la conversation :
    {chat_history}

    Nouvelle question :
    {question}

    Réponds de façon claire, personnalisée, sans répéter ce qui a déjà été dit. Ne redis pas les salutations à chaque message. Sois naturel et pertinent.
    """
    )


    chain = {
        "question": RunnablePassthrough(),
        "context": retriever,
        "chat_history": RunnablePassthrough()
    } | prompt | llm

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

def get_recent_chat_history(chat_history, k=1):
    messages = chat_history.messages[-k*1:]
    return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


# --- Base vectorielle ---
FAISS_INDEX_PATH = "knowledge_faiss-1000"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever()
#retriever = vectordb.as_retriever(search_kwargs={"k": 3})


st.set_page_config(page_title="IA Djom", page_icon="🤖")
st.title("IA Djom - Assistant d'orientation")
chat_history = StreamlitChatMessageHistory()

# --- Sélection du modèle ---
# model_choice = st.selectbox("Choisir un modèle IA", ["llama instant", "Mistral", "openAI", "llama versatile"], key="model_select")
with st.sidebar:
    st.markdown("## Choisir un modèle IA")
    model_choice = st.radio(
        "Modèle à utiliser :",
        ["llama", "llama versatile", "gemma"], key="model_select"
    )
    if st.button("Exporter le chat"):
        if chat_history.messages:
            txt_content = generate_txt(chat_history.messages)
            st.download_button(
                label="📥 Télécharger le TXT",
                data=txt_content,
                file_name=f"chat_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("Aucune conversation à exporter.")
        
        
model_mapping = {
    "llama": "llama-3.1-8b-instant",
    "llama versatile": "llama-3.3-70b-versatile",
    "Mistral": "mistral-saba-24b",
    "gemma": "gemma2-9b-it",
    "qwen": "qwen/qwen3-32b",
    "deepseek": "deepseek-r1-distill-llama-70b", #affiche le thinking
}

selected_model = model_mapping[model_choice]

# --- Gestion de l'historique ---
if len(chat_history.messages) == 0:
    chat_history.add_ai_message(f"Salut !\nJe suis Djom, ton conseiller d'orientation scolaire et professionnelle. Pose-moi une question !")

# --- Affichage des anciens messages ---
for msg in chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)
        
#conseiller("Je suis en terminale scientifique. Que puis-je faire comme études ?")
        

# Interface de chat
user_input = st.chat_input("Demande conseil ici")

if user_input:
    # Affiche message utilisateur
    st.chat_message("user").markdown(user_input)
    start = time.time()
    with st.spinner("conseils..."):
        # Appel du chatbot
        chat_history_text = get_recent_chat_history(chat_history)

        chatbot = get_chain(selected_model)
        response = chatbot.invoke({"question": user_input, "chat_history": chat_history_text}, config={"configurable": {"session_id": "user"}})
        duration = time.time() - start

        # Affiche réponse
        model_name_display = model_choice.replace("-", " ").capitalize()
        response_finale = response.content.strip() + f"\n\n_Réponse générée par le modèle **{model_name_display}** en {duration:.2f} secondes._"
        st.chat_message("ai").markdown(response_finale.strip())
