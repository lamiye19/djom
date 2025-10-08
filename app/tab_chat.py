import streamlit as st
#from langchain_ollama import OllamaEmbeddings
import time
import datetime
from groq import Groq
import uuid
from prompts import *
from app.functions import *
import re
import json


client = Groq(api_key=os.environ["OPENAI_API_KEY"])

def extract_json(response_text):
    match = re.search(r'\{[\s\S]*\}', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("JSON mal formé, tentative de correction…")
            cleaned = match.group().replace("\n", "").replace("**", "")
            return json.loads(cleaned)
    else:
        print("\nErreur:",response_text)
        return {"intention": "erreur", "entities": []}
    
def detect_intent(message):
    try:
        response = client.chat.completions.create(
            model=MODEL_MAP["openai"],
            messages=[
                {"role": "system", "content": PROMPT_CLASSIFICATION.replace("<<< MESSAGE >>>", message)},
            ],
            temperature=0.4,
            max_tokens=800,
        )

        content = response.choices[0].message.content
        predicted = extract_json(content)

        return predicted
    except Exception as e:
            return f"\n\n*(Erreur: {e})*"
    
def detect_profil(message):
    try:
        response = client.chat.completions.create(
            model=MODEL_MAP["gemma"],
            messages=[
                {"role": "system", "content": PROFIL_PROMPT.replace("<<< MESSAGE >>>", message)},
            ],
            temperature=0.4,
            max_tokens=800,
        )

        content = response.choices[0].message.content
        profil = extract_json(content)

        supabase.table("chats").update({
            "resume" : profil.get("resumé"),
            "entities": profil.get("entities")
        }).eq("session_id",st.session_state["session_id"]).execute()

        return profil
    except Exception as e:
            return f"\n\n*(Erreur: {e})*"


# Construire une requête enrichie
def enriched_query(k=3):
        past_questions = [
            m["content"] for m in st.session_state["messages"] 
            if m["role"] == "user"
        ][-k:]
        
        enriched_query = " ".join(past_questions)
        return enriched_query

def filter_docs_with_llm(query, docs):
    # On formate les docs
    docs_text = "\n\n".join([f"[Doc {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    
    prompt = DOC_FILTER_PROMPT\
                .replace("<<< QUESTION >>>", query) \
                .replace("<<< DOCS >>>", docs_text)

    response = client.chat.completions.create(
        model=MODEL_MAP["gemma"],
        messages=[{"role": "system", "content": prompt}]
    )
    
    try:
        filtered = json.loads(response.choices[0].message.content)
        selected = filtered.get("pertinents", [])
        return [docs[i-1] for i in selected if 1 <= i <= len(docs)]
    except Exception as e:
        print("Erreur parsing LLM:", e)
        return docs[:3]
    
# Pour eviter l'erreur 413 (Payload Too Large avec l’API Groq)
def get_context(query, elements,max_chars=1800):
    if elements.get("intention") == "discussion":
        return []
    elif elements.get("intention") == "hors_contexte":
        return ["Le message de l'utilisateur est hors contexte"]
    elif elements.get("intention") == "non_ethique":
        return ["Le message de l'utilisateur ne respecte pas les normes d'éthiques"]
    
    docs = retriever.invoke(query)[:10]
    # docs = retriever.get_relevant_documents(query)[:3]

    if len(docs) == 0 and elements.get("entities"):
        q = " ".join([e['entity'] for e in elements.get("entities")])
        docs = retriever.invoke(q)[:10]
        print("AVEC ENTITY:", q)

    print("\n",len(docs), "avant filtre.\n")
    if len(docs) != 0:
        docs = filter_docs_with_llm(query, docs)
        print("\n",len(docs), "après filtre.\n")
    ctx = "\n\n".join(d.page_content for d in docs)
        
    # Découpage du contexte sans couper brutalement
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars*2, chunk_overlap=50)
    parts = splitter.split_text(ctx)
        
    #return parts[0] if parts else ""
    return docs


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

def user_historique():
    msgs = [m["content"] for m in st.session_state["messages"] if m["role"] == "user"]
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

def reformulation():
        try:
            text = user_historique()
            prompt = REFORMULE_PROMPT \
                .replace("<<< HISTORIQUE_CONVERSATION >>>", "\n".join(text[:-1])) \
                .replace("<<< DERNIER_MESSAGE >>>", text[-1])

            response = client.chat.completions.create(
                model=MODEL_MAP["llama"],
                messages=[{"role": "system", "content": prompt}],
                temperature=0.5,
                max_tokens=800,
            )
            #print("\n".join(text))
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"\n\n*(Erreur: {e})*"

# Sélection du modèle
MODEL_MAP = {
    "llama": "llama-3.1-8b-instant",
    "llama versatile": "llama-3.3-70b-versatile",
    "gemma": "gemma2-9b-it",
    "openai": "openai/gpt-oss-120b"
}


# INTERFACE
def chatbot_tab():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        #st.session_state["session_id"] = "1790e7d4-b2ea-45da-9ad6-746ce7e48bfa"

    # Pour forcer la notation
    if "awaiting_feedback" not in st.session_state:
        st.session_state["awaiting_feedback"] = False

    if "profile" not in st.session_state:
        st.session_state["profile"] = {}

    if "intent" not in st.session_state:
        st.session_state["intent"] = {}


    st.set_page_config(page_title="IA Djom", page_icon="🤖")
    st.title("IA Djom - Assistant d'orientation")

    
    MODEL_ID = MODEL_MAP["llama"]

    with st.sidebar:
        st.caption(f"Modèle Groq: `{MODEL_ID}`")

        st.json(st.session_state["profile"])
        

    # Gestion de l'historique
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
    

    # Interface de chat
    user_input = st.chat_input("Demande ton conseil ici" if not st.session_state["awaiting_feedback"] 
        else "Vous devez d'abord noter la dernière réponse",
        disabled=st.session_state["awaiting_feedback"]
    )


    if user_input:
        if len(st.session_state["messages"]) == 2:
            supabase.table("chats").insert({"session_id":st.session_state["session_id"]}).execute()

        st.session_state["awaiting_feedback"] = True
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Reformulation
        # st.markdown(reformulation())
        with st.expander("Voir les étapes du raisonnement", expanded=True):
            st.caption("Détection de l'intention et des entités")
            intent = detect_intent(user_input)
            if type(intent) is dict:
                st.session_state["intent"] = intent
                st.json(json.dumps(intent, ensure_ascii=False, indent=2))
                if intent.get('entities'):
                    st.caption("Détection du profil utilisateur: ")
                    profil = detect_profil("\n".join(user_historique()))
                    if type(profil) is dict:
                        st.session_state["profile"] = profil
                    else:
                        st.caption(profil)
                    st.caption("Done")
            else:
                st.caption(intent)
            st.caption("Récupération du contexte: ")
        

            # Contexte & messages pour le modèle
            context = get_context(user_input, intent)
            st.caption(f"{len(context)} document(s) récupéré(s)")
            
            st.session_state["contexte"] = "\n\n".join(d.page_content if hasattr(d, "page_content") else str(d) for d in context)
        st.caption("Pause de 5s avant la génération de la réponse ")

        model_messages = build_model_messages(SYSTEM_PROMPT, context, user_input, max_turns=4)
        time.sleep(5)
        start = time.time()
        with st.chat_message("assistant"):
            chunks = stream_completion(model_messages, MODEL_ID)
            answer = st.write_stream(chunks)
            duree = time.time() - start
            caption = f"Réponse générée par **{MODEL_ID}** en {duree:.2f}s."
            st.caption(caption)

            st.session_state["question"] = user_input
            st.session_state["answer"] = answer
            st.session_state["duree"] = duree

            st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()
        

    if "note" not in st.session_state:
        st.session_state["note"] = 3

    if st.session_state["awaiting_feedback"]:
        with st.form("feedback_form"):
                st.slider("Notez cette réponse (1 = mauvaise, 5 = excellente)", 1, 5, key="note")
                submitted = st.form_submit_button("Soumettre la note")
                if submitted:
                    
                    save_feedback(
                        question=st.session_state["question"],
                        reponse=st.session_state["answer"],
                        note=st.session_state["note"],
                        contexte=st.session_state["contexte"],
                        duree=st.session_state["duree"],
                        session_id=st.session_state["session_id"],
                        intention=st.session_state["intent"].get("intention"),
                        entities=st.session_state["intent"].get("entities")
                    )
                    st.success("Merci pour votre feedback !")
                    st.session_state["awaiting_feedback"] = False
                    st.rerun()
                 
        