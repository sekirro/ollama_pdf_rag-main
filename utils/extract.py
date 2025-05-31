import os
import chardet
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

def process_folder(folder_path):
    documents = []
    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.txt': TextLoader,
        # '.md': TextLoader,
        # '.markdown': TextLoader
    }
    
    # 为代码文件单独处理
    code_extensions = {'.py', '.java', '.js', '.cpp', '.c', '.h', '.hpp', '.md', '.markdown'}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            try:
                if file_extension in supported_extensions:
                    # 对于支持的文件类型，先检测编码
                    if file_extension in {'.txt', '.csv'}:  # 只对文本类型的文件检测编码
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                            detected = chardet.detect(raw_data)
                            encoding = detected['encoding']
                        
                        loader_class = supported_extensions[file_extension]
                        if file_extension == '.csv':
                            loader = loader_class(file_path, encoding=encoding)
                        else:
                            loader = loader_class(file_path, encoding=encoding)
                    else:
                        # 对于非文本类型的文件（如pdf, docx等）使用默认加载器
                        loader_class = supported_extensions[file_extension]
                        loader = loader_class(file_path)
                    docs = loader.load()
                    
                elif file_extension in code_extensions:
                    # 对代码文件特殊处理
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        detected = chardet.detect(raw_data)
                        encoding = detected['encoding']
                    loader = TextLoader(file_path, encoding=encoding)
                    docs = loader.load()
                else:
                    continue

                # 添加额外的元数据
                for doc in docs:
                    file_path = os.path.relpath(file_path, folder_path)
                    formatted_content = f"""
文件名: {file}
文件路径: {file_path}
文件类型: {file_extension}
---
{doc.page_content}
"""
                    doc.page_content = formatted_content
                    doc.metadata.update({
                        'filename': file,
                        'extension': file_extension,
                        'relative_path': file_path
                    })
                documents.extend(docs)
                print(f"成功处理文件: {file}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return documents

# 使用修改后的函数
def transform_documents(folder_path):
    # 直接返回处理后的文档列表
    return process_folder(folder_path)

if __name__ == "__main__":
    folder_path = "D:/Desktop/ollama_pdf_rag-main/documents_for_analyse"
    documents = transform_documents(folder_path)
    print(documents)
