import streamlit as st
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import datetime
from groq import Groq
from app.train import detect_intention

def chatbot_nlu_tab():
    client = Groq(api_key=os.environ["OPENAI_API_KEY"])

    # --- Base vectorielle ---
    FAISS_INDEX_PATH = "knowledge_faiss"
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_kwargs={"device": "cpu"}
    )
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7}
    )

    GREETINGS = {"salut", "bonjour", "bonsoir", "hello", "hi", "slt", "bjr", "comment vas tu?", "djom", "ok", "d'accord", "merci"}

    def is_greeting(q: str) -> bool:
        ql = q.strip().lower()
        return any(ql.startswith(g) or ql == g for g in GREETINGS)


    # Pour eviter l'erreur 413 (Payload Too Large avec l’API Groq)
    def get_context(query, max_chars=1800):
        intent = detect_intention(query)

        if is_greeting(query):
            intent = "salutation"
            return ""
        
        st.session_state["intentions"].append({
            "entree" : query,
            "intention": intent
        })


        if intent == "hors_contexte":
            return ""  # pas de recherche
        elif intent == "recherche_centre":
            docs = retriever.invoke(query + " centre formation")[:3]
        elif intent == "recherche_formation":
            docs = retriever.invoke(query + " formation université")[:3]
        elif intent == "recherche_emploi":
            docs = retriever.invoke(query + " emploi métier")[:3]
        elif intent == "recherche_info":
            docs = retriever.invoke(query, k=5)
        if intent == "info_interet":
            st.session_state["user_profile"].append(query)
            docs = retriever.invoke(query)[:3]
        else:
            docs = retriever.invoke(query)[:3]
        # docs = retriever.invoke(query)[:3]

        ctx = "\n\n".join(d.page_content for d in docs)
        
        # Découpage du contexte sans couper brutalement
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=50)
        parts = splitter.split_text(ctx)
        return parts[0] if parts else ""
        
    def generate_txt():
        if "messages" in st.session_state and st.session_state["messages"]:
            lines = []
            for m in st.session_state["messages"]:
                role = "Utilisateur" if m["role"] == "user" else "Assistant"
                lines.append(f"{role} : {m['content']}")
            txt = "\n\n".join(lines)
            return txt
        
    def get_recent_chat_history(chat_history, k=1):
        messages = chat_history.messages[-k*1:]
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])


    #
    # INTERFACE
    #
    st.set_page_config(page_title="IA Djom", page_icon="🤖")
    st.title("IA Djom - Assistant d'orientation")

    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = []

    if "intentions" not in st.session_state:
        st.session_state["intentions"] = []

    # --- Sélection du modèle ---
    # model_choice = st.selectbox("Choisir un modèle IA", ["llama instant", "Mistral", "openAI", "llama versatile"], key="model_select") 

    with st.sidebar:
        MODEL_ID = "llama-3.1-8b-instant"
        st.caption(f"Modèle Groq: `{MODEL_ID}`")

        if st.session_state["user_profile"]:
            st.markdown("## PROFIL Utilisateur")
            st.json({
                "PROFIL": st.session_state["user_profile"]
            })

        if st.session_state["intentions"]:
            st.markdown("## Historique")
            st.json({
                0 : st.session_state["intentions"]
            })
            


    # --- Gestion de l'historique ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Message d’accueil
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Salut !\nJe suis Djom. Pose-moi une question d’orientation !"
        })
        
    for m in st.session_state["messages"]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    # Limiter l’historique envoyé au modèle (évite 413)
    def build_model_messages(system_prompt, context, new_user_msg, max_turns=4):
        recent = [m for m in st.session_state["messages"] if m["role"] in ("user","assistant")]
        trimmed = recent[-(max_turns*2):]

        msgs = [{"role": "system", "content": system_prompt}]
        if context:
            msgs.append({"role": "system", "content": f"Contexte (extraits):\n{context}"})
        if st.session_state.get("user_profile"):
            profile = st.session_state["user_profile"]
            profile_txt = ', '.join(profile)
            msgs.append({"role": "system", "content": f"Profil utilisateur connu : {profile_txt}"})
            
        # Ajouter l’historique
        for m in trimmed:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": new_user_msg})
        return msgs
    
    SYSTEM_PROMPT = (
        "Tu es Djom, un conseiller d’orientation scolaire et professionnelle au Togo. "
        "Réponds clairement, de manière personnalisée, sans répéter ce qui a déjà été dit. "
        "Sois naturel et appuie-toi sur le contexte fourni s’il est pertinent.\n"
        "Règles :\n"
        "- Si le message est un simple salut ou trop vague, demande une clarification (classe/niveau, intérêts).\n"
        "- N’évoque pas d’offres d’emploi sauf si l’utilisateur en parle explicitement.\n"
        "- Réponds clairement et sans répéter les salutations à chaque message."
    )

    # Générateur (reponses) pour st.write_stream
    def stream_completion(messages, model_id):
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.3,
                max_tokens=800,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                yield delta
        except Exception as e:
            yield f"\n\n*(Erreur: {e})*"

    # Interface de chat
    user_input = st.chat_input("Demande ton conseil ici")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Contexte & messages pour le modèle
        context = get_context(user_input)
        model_messages = build_model_messages(SYSTEM_PROMPT, context, user_input, max_turns=4)

        start = time.time()
        with st.chat_message("assistant"):
            chunks = stream_completion(model_messages, MODEL_ID)
            answer = st.write_stream(chunks)
            dur = time.time() - start
            caption = f"Réponse générée par **{MODEL_ID}** en {dur:.2f}s."
            st.caption(caption)

            st.session_state["messages"].append({"role": "assistant", "content": answer, "caption": caption})