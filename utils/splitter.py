from langchain_text_splitters import RecursiveCharacterTextSplitter
import utils.extract as extract

def split_text(data, chunk_size=7500, chunk_overlap=100):
    '''
    输入：data，documents类型
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def main():
    local_path = "D:/Desktop/ollama_pdf_rag-main/documents_for_analyse"
    data = extract.transform_documents(local_path)
    chunks = split_text(data)
    print(chunks)

if __name__ == "__main__":
    main()
