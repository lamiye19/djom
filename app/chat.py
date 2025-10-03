import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import datetime
import time
from groq import Groq

def chatbot_tab():

    # ===========================
    # --- INITIALISATION ---
    # ===========================

    client = Groq(api_key=os.environ["OPENAI_API_KEY"])

    # Embeddings communs
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"}
    )

    # Chargement des vectordb
    docs_vectorstore = FAISS.load_local("knowledge_faiss", embeddings, allow_dangerous_deserialization=True)
    # Crée la mémoire chat si elle n'existe pas
    if os.path.exists("chat_memory_faiss"):
        chat_memory_store = FAISS.load_local("chat_memory_faiss", embeddings, allow_dangerous_deserialization=True)
    else:
        chat_memory_store = FAISS.from_texts([""], embeddings)

    # Paramètres
    K_TOP = 5  # nombre de passages récupérés
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    GREETINGS = {"salut", "bonjour", "bonsoir", "hello", "hi", "slt", "bjr", "comment vas tu?", "djom", "ok", "d'accord", "merci"}

    # ===========================
    # --- FONCTIONS ---
    # ===========================

    # Construire une requête enrichie
    def enriched_query(k=3):
        past_questions = [
            m["content"] for m in st.session_state["messages"] 
            if m["role"] == "user"
        ][-k:]
        
        enriched_query = " ".join(past_questions)
        return enriched_query

    def is_greeting(q: str) -> bool:
        ql = q.strip().lower()
        return any(ql.startswith(g) or ql == g for g in GREETINGS)

    def get_hybrid_context(user_input):
        """Récupère contexte des documents + historique"""
        if is_greeting(user_input):
            return ""
        query = enriched_query()
        print("Question enrichie:", query )
        
        # Documents
        doc_results = docs_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7}
        ).invoke(user_input)
        print("doc:", doc_results)
        # Historique chat
        mem_results = chat_memory_store.as_retriever(
            search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7}
        ).invoke(user_input)
        print("\nHist:", doc_results)
        
        # Fusion
        combined_context = "\n\n".join([d.page_content for d in doc_results + mem_results])
        return combined_context[:3000]  # limite pour éviter 413 payload

    def update_chat_memory(user_input, bot_response):
        """Ajoute l'échange à la mémoire vectorielle"""
        text = f"Utilisateur: {user_input}\nAssistant: {bot_response}"
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        chat_memory_store.add_texts(chunks)
        chat_memory_store.save_local("chat_memory_faiss")

    def generate_txt():
        if "messages" in st.session_state and st.session_state["messages"]:
            lines = []
            for m in st.session_state["messages"]:
                role = "Utilisateur" if m["role"] == "user" else "Assistant"
                lines.append(f"{role} : {m['content']}")
            txt = "\n\n".join(lines)
            return txt

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

    def build_model_messages(system_prompt, context, user_input, max_turns=4):
        recent = [m for m in st.session_state["messages"] if m["role"] in ("user","assistant")]
        trimmed = recent[-(max_turns*2):]

        msgs = [{"role": "system", "content": system_prompt}]
        if context:
            msgs.append({"role": "system", "content": f"Contexte (documents + historique):\n{context}"})
            
        for m in trimmed:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user_input})
        return msgs

    # ===========================
    # --- STREAMLIT UI ---
    # ===========================

    st.set_page_config(page_title="IA Djom", page_icon="🤖")
    st.title("IA Djom - Assistant d'orientation")

    # Sidebar export & modèle
    with st.sidebar:
        st.markdown("## Exporter le chat")
        if st.button("Exporter la conversation"):
            txt_content = generate_txt()
            if txt_content:
                st.download_button(
                    label="Télécharger le TXT",
                    data=txt_content.encode("utf-8"),
                    file_name=f"chat_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Aucune conversation à exporter.")

    # Initialisation session
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Salut !\nJe suis Djom. Pose-moi une question d’orientation !"
        })

    for m in st.session_state["messages"]:
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])

    # SYSTEM PROMPT strict
    SYSTEM_PROMPT = (
        "Tu es Djom, conseiller d’orientation scolaire et professionnelle au Togo. "
        "Réponds uniquement à partir du contexte fourni (documents + historique). "
        #"Si l’information n’est pas dans le contexte, dis simplement que tu ne sais pas. "
        "Ne brode pas, ne devine pas, reste clair et précis. "
        "Ne répète pas les salutations."
    )

    # Chat input
    user_input = st.chat_input("Pose ta question ici")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        context = get_hybrid_context(user_input)
        model_messages = build_model_messages(SYSTEM_PROMPT, context, user_input, max_turns=4)

        start = time.time()
        with st.chat_message("assistant"):
            chunks = stream_completion(model_messages, model_id="llama-3.1-8b-instant")
            answer = st.write_stream(chunks)
            dur = time.time() - start
            st.session_state["messages"].append({"role": "assistant", "content": answer})
        
        update_chat_memory(user_input, answer)
