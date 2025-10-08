from supabase import create_client
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
supabase = create_client(url, key)
FAISS_INDEX_PATH = "knowledge_faiss"
embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"}
)
vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.7
    }
)


def save_feedback(question, reponse, note, contexte, duree, session_id, intention, entities):
    data = {
        "session_id": session_id,
        "question": question,
        "reponse": reponse,
        "note": note,
        "contexte": contexte,
        "duree": duree,
        "intention": intention,
        "entities": entities
    }
    supabase.table("messages").insert(data).execute()



        
