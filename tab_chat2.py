import streamlit as st
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import json
import datetime
from groq import Groq
from prompts import PROMPT_CLASSIFICATION

def classif():
    client = Groq(api_key=os.environ["OPENAI_API_KEY"])
    prompt = PROMPT_CLASSIFICATION

    def stream_completion(messages):
        try:
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": messages}],
                temperature=0.3,
                max_tokens=800,
                stream=False,
            )

            data = stream.choices[0].message.content
            
            try:
                data = json.loads(data)
                #data = json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                print("\nErreur : la sortie n'est pas un JSON valide.")

            yield data
        except Exception as e:
            yield f"\n\n*(Erreur: {e})*"

    message = "J’ai eu un bac scientifique avec 15.45 comme moyenne et je veux m’orienter vers les sciences et ingénieries."
    message2 = "conseille moi"

    response = stream_completion(prompt.replace("<<< MESSAGE UTILISATEUR >>>", message))
    answer = st.write_stream(response["intention"])

        

def chatbot_tab():
    client = Groq(api_key=os.environ["OPENAI_API_KEY"])

    # --- Base vectorielle ---
    FAISS_INDEX_PATH = "knowledge_faiss"
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
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


    #
    # INTERFACE
    #
    st.set_page_config(page_title="IA Djom", page_icon="🤖")
    st.title("IA Djom - Assistant d'orientation")

    # --- Sélection du modèle ---
    # model_choice = st.selectbox("Choisir un modèle IA", ["llama instant", "Mistral", "openAI", "llama versatile"], key="model_select") 
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
            
        # Ajouter l’historique
        for m in trimmed:
            msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": new_user_msg})
        return msgs
    
    SYSTEM_PROMPT = (
        "Tu es Djom, un conseiller d’orientation scolaire et professionnelle au Togo. "
        "Réponds clairement, de manière personnalisée, sans répéter ce qui a déjà été dit. "
        "Sois naturel et appuie-toi uniquement sur le contexte fourni.\n"
        "Règles :\n"
        "- Si le message est un simple salut ou trop vague, demande une clarification (classe/niveau, intérêts).\n"
        "- N’évoque pas d’offres d’emploi sauf si l’utilisateur en parle explicitement.\n"
        "- Réponds qu'aux questions ayant rapport avec l'orientation scolaire et professionnelle.\n"
        "- Reste dans le contexte togolais.\n"
        "- Pour les questions hors cadre d'orientation, réponds brièvement en disant que tu n'est pas programmé pour ça.\n"
        "- Réponds clairement et sans répéter les salutations à chaque message."
    )

    SYSTEM_PROMPT = (
    "Tu es Djom, un conseiller d’orientation scolaire et professionnelle au Togo. "
    "Réponds uniquement en utilisant le contexte fourni et les informations historique. "
    "Si le message est trop vague ou il n'y a pas d'informations sur le profil de la personne, demande une clarification (classe/niveau, intérêts).\n"
    "Si l’information n’est pas dans le contexte, dis simplement que tu n'as pas connaissance de cette information. "
    "Reste dans le contexte togolais et de l'orientation scolaire et professionnelle.\n"
    "Ne devine pas, ne brode pas, reste clair et précis. "
    "Ne répète pas les salutations."
    "Si le message est hors sujet.\n"
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