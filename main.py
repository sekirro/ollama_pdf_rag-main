## 需要开ollama，命令行运行ollama list就行

import utils.extract as extract
import utils.splitter as splitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
import torch
import chromadb
from langchain.schema import Document

def load_data():
    local_path = "D:/Desktop/ollama_pdf_rag-main/documents_for_analyse"
    data = extract.transform_documents(local_path)
    return data

def split_data(data):
    chunks = splitter.split_text(data)
    return chunks

def load_embeddings(flag=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if flag:
        # embeddings_path = 'intfloat/multilingual-e5-large-instruct'
        # embeddings_path = "shibing624/text2vec-base-chinese"
        embeddings_path = "BAAI/bge-large-zh-v1.5"
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_path,
            model_kwargs={
                'device': device,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
    else:
        embeddings_path = "nomic-embed-text"
        embeddings = OllamaEmbeddings(model=embeddings_path)

    return embeddings

def create_vector_db(chunks, embeddings):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="local-rag"
    )
    return vector_db

def create_retriever(vector_db):
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5,
        }
    )
    return retriever

def create_chain(retriever, llm):
    template = """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明。

上下文信息：
{context}

问题：{question}

回答："""

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)   
    return chain


def main():
    local_model = "deepseek-r1"
    # local_model = "qwen2.5"
    llm = ChatOllama(model=local_model)

    data = load_data()
    chunks = split_data(data)

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = load_embeddings()
    vector_db = create_vector_db(chunks, embeddings)
    retriever = create_retriever(vector_db)
    # question = input("请输入问题：")
    # while question != "exit":
    #     documents = retriever.invoke(question)
    #     # 提取并打印文件名
    #     for doc in documents:
    #         filename = doc.metadata['filename']
    #         print(filename)
    #     question = input("请输入问题：")
    chain = create_chain(retriever, llm)
    question = input("请输入问题：")
    while question != "exit":
        result = chain.invoke(question)
        print(result)
        question = input("请输入问题：")


    

if __name__ == "__main__":
    main()