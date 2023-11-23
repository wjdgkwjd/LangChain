from dotenv import load_dotenv 

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import asyncio

import pdfplumber
from collections import namedtuple

load_dotenv()


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tiktoken_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens )

# loader = PyPDFLoader("files\jshs-history.pdf")
# documents = loader.load() 
Document = namedtuple('Document', ['page_content', 'metadata'])
documents =[]
with pdfplumber.open("files/jshs-history.pdf") as pdf_document:
    for page_number, page in enumerate(pdf_document.pages):
        text = page.extract_text()

        metadata = {
            'source': 'files/jshs-history.pdf',
            'page': page_number + 1
        }
        document = Document(page_content=text, metadata=metadata)
        documents.append(document)
        
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =50,
        chunk_overlap  = 0,
        separators=["\n"],
        length_function =tiktoken_len
    )

pages = text_splitter.split_documents(documents)
print( len(pages) )
i=0
for p in pages:
    i=i+1
    print( "{:02d} {:02d} ".format(i, tiktoken_len(p.page_content)), p.page_content, p.metadata['source'])

print("="*00)
index = FAISS.from_documents(pages , OpenAIEmbeddings())

index.save_local("faiss-jshs-history-pdf")

query = "현재 교장은?"
# docs = index.similarity_search(query) 유사도가 없다.
loop = asyncio.get_event_loop()
docs = loop.run_until_complete( index.asimilarity_search_with_relevance_scores(query) ) # 유사도 있는 비동기 개체호출 

print(query +"  >> 답변에 사용할 문장 문장 검색 ")

print("-"*100)

for doc, score in docs:
    print(f"{score}\t{doc.page_content}")

print("="*00)

from langchain.chat_models import ChatOpenAI

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

chain = load_qa_chain(llm_model, verbose=False)

query = "현재 교장은 ? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)

query = "1회 졸업 인원수 ? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "초대 교장은? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)