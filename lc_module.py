import time
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from setting_inf import *

def doc2vector(url,embeddings= embeddings):
    """
    
    Parameters:
    - url (str): 文本網址/PDF/字串
    - embeddings ("textembedding-gecko@001"): VertexAIEmbeddings
    return:
    - db (langchain_community.vectorstores.chroma.Chroma): 轉為向量的資料
      
    """

    if url[-4:]=='html':
        loader = WebBaseLoader(url)
        documents = loader.load()
    elif url[-3:] == 'pdf':
        loader = PyPDFLoader(url)
        documents = loader.load()
 
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    docs = text_splitter.split_documents(documents)
    
    """Store docs in local vectorstore as index
       it may take a while since API is rate limited
    """
    db = Chroma.from_documents(docs, embeddings)
    return db

def retrieve(db,llm,query):
    """   
    Parameters:
    - db (langchain_community.vectorstores.chroma.Chroma): 轉為向量的資料
    - llm ("text-bison@001"): 
    - query (str): 查詢字串

    return:
    - dict: 包含 'query' 和 'result' 
        'query' 為輸入 'result' langchain 輸出結果      
    """
    
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Uses LLM to synthesize results from the search index.
    # We use Vertex PaLM Text API for LLM
    qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    result = qa({"query": query})
    return result

if __name__ == '__main__' :    
    url = 'http_5.pdf'    
    db = doc2vector(url)
    query = "when will system occur 511"
    result = retrieve(db,llm,query)  
    print(result['result'])

