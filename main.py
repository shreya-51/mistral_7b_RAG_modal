from modal import Image, Stub, NetworkFileSystem

from langchain.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from helpers import scrape_text, load_documents, split_documents, embed_docs

volume = NetworkFileSystem.new() # non persisted
image = (Image.from_registry("python:3.11-slim-bookworm").pip_install(
    "beautifulsoup4~=4.11.1",
    "langchain~=0.0.138",
    "openai~=0.27.4",
    "tiktoken==0.3.0",
    "requests",
    "unstructured",
    "sentence-transformers",
    "chromadb",
    "vllm",   
))

stub = Stub("testing-rag", image=image)

@stub.function(gpu="A100", network_file_systems={"/scrape": volume})
def run(query):
    print("Setup started.\n")
    
    # step 1
    dir = scrape_text()
    print("1/4: Finished scraping documents.\n")
    print(f"Scraped directory: {dir}")

    # step 2
    docs = load_documents(dir)
    print("2/4: Finished loading documents.\n")
    print(f"Number of documents loaded: {len(docs)}")

    # step 3
    docs_split = split_documents(docs)
    print("3/4: Finished splitting documents.\n")
    print(f"Number of documents after chunking: {len(docs_split)}")

    # step 4
    retriever = embed_docs(docs_split)
    print("4/4: Finished embedding documents.\n")

    print("Setup finished.")
    
    # set up LLM w pagedattention
    llm = VLLM(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        trust_remote_code=True,
        max_new_tokens=500,
        top_k=10,
        top_p=0.95,
        temperature=0.5,
    )
    
    # prompt template 
    qa_template = """<s>[INST] You are a helpful assistant.
    Use the following context to Answer the question below briefly:

    {context}

    {question} [/INST] </s>
    """
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    # custom QA Chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    response = qa_chain({"query": query})
    
    # llm answer
    print("LLM Answer:\n")
    print(response['result'])

    print("\nRetreived source documents:\n")
    print(response['source_documents'])

@stub.local_entrypoint()
def main():
    question = "How do I use langchain with modal?"
    run.remote(question)
    