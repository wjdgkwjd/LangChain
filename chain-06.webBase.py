from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 
import json
import re

from langchain.document_loaders import WebBaseLoader,UnstructuredURLLoader
import _myfunction as myfun
loader = WebBaseLoader(web_path="https://jeju-s.jje.hs.kr/jeju-s/0102/history")
# loader = WebBaseLoader(web_path="https://api.salvion.kr/of=T10&sc=9290066&ac=date&sd=2023-10-04&ed=2023-10-05&code=all")

documents = loader.load()
print( documents  )
print( "="*100 )
# quit()
# parsed_content = json.loads(documents[0].page_content)
# documents[0].page_content=re.sub(r'\s+', ' ', documents[0].page_content).strip()



from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap=0,
        separators=["\n\n"],
        length_function =myfun.tiktoken_len
    )
pages = text_splitter.split_documents(documents)
print( len(pages) )

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

index = FAISS.from_documents(pages , OpenAIEmbeddings())

index.save_local("faiss-web")
llm_model = ChatOpenAI(model_name="gpt-4", temperature=0)  

chain = load_qa_chain(llm_model, verbose=False)

query = "제주과학고등학교 교장 ? "
docs = index.similarity_search(query)

res = chain.run(input_documents=docs, question=query)
print( query,res)


query = "교무실 전화번호 ? "
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query,res)
