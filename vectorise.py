import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_split_txt(txt_dir="../collecte2/data"):
    """
    Charge tous les fichiers TXT depuis un dossier et les découpe en chunks.
    """
    all_docs = []
    paths = []
    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                paths.append(path)
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                all_docs.extend(chunks)
    print(len(paths), "fichiers")
    return all_docs
    
def create_vectorstore(documents, index_path="knowledge_faiss"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"})
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    index_file = os.path.join(index_path, "index.faiss")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    print("Base vectorielle créee et contient", len(vectorstore), "documents")
    return vectorstore
    
txt = load_and_split_txt()

vectorstore = create_vectorstore(txt)
