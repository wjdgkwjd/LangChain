from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain



import asyncio
from dotenv import load_dotenv 

load_dotenv() # openai_key  .env 선언 사용 

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tiktoken_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


loader = TextLoader("files/unsu.txt", encoding='utf-8')
documents = loader.load()
# documents[0].page_content=documents[0].page_content.replace("\n"," ")

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =100,
        chunk_overlap  = 0,
        separators=[". "],
        length_function =tiktoken_len
    )

pages = text_splitter.split_documents(documents)

print( len(pages) )
i=0
for p in pages:
    i=i+1
    print( "{:02d} {}".format(i, tiktoken_len(p.page_content)), p.page_content.replace('\n', ''), p.metadata['source'])

print("="*100)

index = FAISS.from_documents(pages, OpenAIEmbeddings())

index.save_local("faiss-unsu-txt")

#query = "아내가 먹고 싶어하는 음식은?"
#loop = asyncio.get_event_loop()
#docs = loop.run_until_complete( index.asimilarity_search_with_relevance_scores(query) )
#for doc, score in docs:
#    print(f"{score}\t{doc.page_content}")

#print("-"*100)


from langchain.chat_models import ChatOpenAI

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

chain = load_qa_chain(llm_model, chain_type="stuff")


query = "아내가 먹고 싶어하는 음식은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query , res)

query = "주인공의 직업은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query ,res)

query = "지은이 ?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query , res)

query = "주인공 친구의 이름은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query, res)

#pdf 사용법 알았다.