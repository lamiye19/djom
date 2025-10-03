import streamlit as st
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import datetime
from groq import Groq
from supabase import create_client
import uuid


DB_PATH = "chat_feedback.sqlite"

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
supabase = create_client(url, key)

def save_feedback(question, reponse, note, contexte, duree, session_id):
    data = {
        "session_id": session_id,
        "question": question,
        "reponse": reponse,
        "note": note,
        "contexte": contexte,
        "duree": duree,
    }
    supabase.table("chat_feedback").insert(data).execute()


GREETINGS = {"salut", "bonjour", "bonsoir", "hello", "hi", "slt", "bjr", "comment vas tu?", "djom", "ok", "d'accord", "merci"}

def is_greeting(q: str) -> bool:
        ql = q.strip().lower()
        return any(ql.startswith(g) or ql == g for g in GREETINGS)

# Pour eviter l'erreur 413 (Payload Too Large avec l’API Groq)
def get_context(query, max_chars=1800):
        if is_greeting(query):
            return ""
        query = enriched_query()
        print("Question enrichie:", query )

        docs = retriever.invoke(query)[:3]
        # docs = retriever.get_relevant_documents(query)[:3]
        ctx = "\n\n".join(d.page_content for d in docs)
        print(ctx,"\n")
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
    
    
    # Construire une requête enrichie
def enriched_query(k=3):
        past_questions = [
            m["content"] for m in st.session_state["messages"] 
            if m["role"] == "user"
        ][-k:]
        
        enriched_query = " ".join(past_questions)
        return enriched_query


SYSTEM_PROMPT = (
    "Tu es Djom, un conseiller d’orientation scolaire et professionnelle au Togo. "
    "Ta mission est de guider les élèves dans leurs choix de filières et de carrières en fonction de leurs performances et aspirations."
    "Réponds uniquement en utilisant le contexte fourni et les informations historique. "
    "Si le message est trop vague ou il n'y a pas d'informations sur le profil de la personne, demande une clarification (classe/niveau, intérêts).\n"
    "Si l’information n’est pas dans le contexte, dis simplement que tu n'as pas connaissance de cette information. "
    "Reste dans le contexte togolais et de l'orientation scolaire et professionnelle.\n"
    "Ne devine pas, ne brode pas, reste clair et précis. "
    "Ne répète pas les salutations."
    "Si le message est hors sujet.\n"
)
    # Limiter l’historique envoyé au modèle (évite 413)
def build_model_messages(system_prompt, context, new_user_msg, max_turns=4):
        recent = [m for m in st.session_state["messages"] if m["role"] in ("user","assistant")]
        trimmed = recent[-(max_turns*2):]



        msgs = [{"role": "system", "content": system_prompt}]
        if context:
            msgs.append({"role": "system", "content": f"Contexte (extraits):\n{context}"})
            
        # Ajouter l’historique
        for m in trimmed:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": new_user_msg})
        return msgs
    

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


client = Groq(api_key=os.environ["OPENAI_API_KEY"])

    # --- Base vectorielle ---
FAISS_INDEX_PATH = "knowledge_faiss"
embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"}
    )
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7}
    )

def chatbot_tab():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    #
    # INTERFACE
    #
    st.set_page_config(page_title="IA Djom", page_icon="🤖")
    st.title("IA Djom - Assistant d'orientation")

    # --- Sélection du modèle ---
    MODEL_MAP = {
        "llama": "llama-3.1-8b-instant",
        "llama versatile": "llama-3.3-70b-versatile",
        "gemma": "gemma2-9b-it",
    }
    MODEL_ID = MODEL_MAP["llama"]

    with st.sidebar:
        st.markdown("## Choisir un modèle IA")
        st.caption(f"Modèle Groq: `{MODEL_ID}`")
        if st.button("Exporter le chat"):
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
            


    # --- Gestion de l'historique ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        # Message d’accueil
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Salut !\nJe suis Djom. Pose-moi une question d’orientation !"
        })
    
    if "awaiting_feedback" not in st.session_state:
        st.session_state["awaiting_feedback"] = False
        
    for m in st.session_state["messages"]:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])

    # Interface de chat
    user_input = st.chat_input("Demande ton conseil ici" if not st.session_state["awaiting_feedback"] else "Vous devez d'abord noter la dernière réponse",
    disabled=st.session_state["awaiting_feedback"])


    if user_input:
        st.session_state["awaiting_feedback"] = True
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Contexte & messages pour le modèle
        context = get_context(user_input)
        st.session_state["contexte"] = context
        model_messages = build_model_messages(SYSTEM_PROMPT, context, user_input, max_turns=4)

        start = time.time()
        with st.chat_message("assistant"):
            chunks = stream_completion(model_messages, MODEL_ID)
            answer = st.write_stream(chunks)
            dur = time.time() - start
            caption = f"Réponse générée par **{MODEL_ID}** en {dur:.2f}s."
            st.caption(caption)

            st.session_state["question"] = user_input
            st.session_state["answer"] = answer
            st.session_state["duree"] = dur

            st.session_state["messages"].append({"role": "assistant", "content": answer, "caption": caption})
        st.rerun()
        

    if "note" not in st.session_state:
        st.session_state["note"] = 3

    if st.session_state["awaiting_feedback"]:
        with st.form("feedback_form"):
                st.slider("Notez cette réponse (1 = mauvaise, 5 = excellente)", 1, 5, key="note")
                submitted = st.form_submit_button("Soumettre la note")
                if submitted:
                    
                    save_feedback(
                        st.session_state["question"],
                        st.session_state["answer"],
                        st.session_state["note"],
                        contexte=st.session_state["contexte"],
                        duree=st.session_state["duree"],
                        session_id=st.session_state["session_id"]
                    )
                    st.success("Merci pour votre feedback !")
                    st.session_state["awaiting_feedback"] = False
                    st.rerun()
                 
        