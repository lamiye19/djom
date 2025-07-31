import streamlit as st
from langchain_community.vectorstores import FAISS
#from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
import time
import datetime
from groq import Groq

client = Groq(api_key=os.environ["OPENAI_API_KEY"])

# --- Base vectorielle ---
FAISS_INDEX_PATH = "knowledge_faiss"
#embeddings = OllamaEmbeddings(model="mxbai-embed-large")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"})
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3}
)

GREETINGS = {"salut", "bonjour", "bonsoir", "hello", "hi", "slt", "bjr", "comment vas tu?", "djom", "ok", "d'accord", "merci"}

def is_greeting(q: str) -> bool:
    ql = q.strip().lower()
    return (ql in GREETINGS) or (len(ql) <= 30 and any(g in ql for g in GREETINGS))

# Pour eviter l'erreur 413
def get_context(query, max_chars=1800):
    if is_greeting(query):
        return ""
    docs = retriever.invoke(query)[:3]
    # docs = retriever.get_relevant_documents(query)[:3]
    ctx = "\n\n".join(d.page_content for d in docs)
    return ctx[:max_chars]
    
def generate_txt():
    lines = []
    if "messages" in st.session_state and st.session_state["messages"]:
            lines = []
            for m in st.session_state["messages"]:
                role = "Utilisateur" if m["role"] == "user" else "Assistant"
                lines.append(f"{role} : {m['content']}")
                if m.get("caption"):
                    lines.append(m["caption"])
            txt = "\n\n".join(lines)
            return txt
    
def get_recent_chat_history(chat_history, k=1):
    messages = chat_history.messages[-k*1:]
    return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])



st.set_page_config(page_title="IA Djom", page_icon="🤖")
st.title("IA Djom - Assistant d'orientation")

# --- Sélection du modèle ---
# model_choice = st.selectbox("Choisir un modèle IA", ["llama instant", "Mistral", "openAI", "llama versatile"], key="model_select") 
MODEL_MAP = {
    "llama": "llama-3.1-8b-instant",
    "llama versatile": "llama-3.3-70b-versatile",
    "Mistral": "mistral-saba-24b",
    "gemma": "gemma2-9b-it",
    "qwen": "qwen/qwen3-32b",
    "deepseek": "deepseek-r1-distill-llama-70b", #affiche le thinking
}

with st.sidebar:
    st.markdown("## Choisir un modèle IA")
    model_choice = st.radio(
        "Modèle à utiliser :",
        ["llama", "llama versatile", "gemma"], key="model_select", index=0
    )
    MODEL_ID = MODEL_MAP[model_choice]
    st.caption(f"Modèle Groq: `{MODEL_ID}`")
    if st.button("Exporter le chat"):
        txt_content = generate_txt().encode("utf-8")
        if txt_content is not None:
            st.download_button(
                label="📥 Télécharger le TXT",
                data=txt_content,
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
    "Réponds clairement, personnalisé, sans répéter ce qui a deja été dit. Sois naturel et appuie-toi sur le contexte fourni et les éléments deja évoqués dans la conversation. Il faut savoir quand s'arreter.\n"
    "Règles :\n"
    "- Si le message est un simple salut ou trop vague, demande une clarification (classe/niveau, intérêts) et n’utilise pas le contexte.\n"
    "- N'évoque pas d’offres d’emploi sauf si l’utilisateur en parle explicitement.\n"
    "- Utilise le contexte uniquement s’il est pertinent.\n"
    "- Réponds clairement, sans répéter les salutations à chaque message."
)

# Générateur pour st.write_stream
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