import os
import requests
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin
from datetime import datetime
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

def scrape_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.content, 'html.parser')
    else:
        print(f"Failed to retrieve page: {url}")
        return None

def get_sidebar_links(soup):
    sidebar_links = []
    for link in soup.find_all("a"):
        href = link.get('href')
        if href and href.startswith('/docs/'):
            full_url = urljoin("https://modal.com", href)
            sidebar_links.append(full_url)
    return sidebar_links

def extract_text(soup):
    # remove script and style elements
    for script_or_style in soup(['script', 'style', 'header', 'footer']):
        script_or_style.extract()

    # remove comments
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # get text
    text = soup.get_text(separator='\n', strip=True)
    return text

def save_text(content, filename, directory):
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
        file.write(content)

def scrape_text():
    tabs = [
        "https://modal.com/docs/examples",
        "https://modal.com/docs/guide",
        "https://modal.com/docs/reference"
    ]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory = f"/scrape/scraped_text_{timestamp}"
    os.makedirs(directory, exist_ok=True)

    for tab_url in tabs:
        tab_soup = scrape_page(tab_url)
        if tab_soup:
            sidebar_links = get_sidebar_links(tab_soup)
            for link in sidebar_links:
                page_soup = scrape_page(link)
                if page_soup:
                    text_content = extract_text(page_soup)
                    filename = link.split('/')[-1] + ".txt"
                    save_text(text_content, filename, directory)
                    # print(f"Saved text of {link} as {filename}") # comment for cleaner logs
    return directory

# load txt files into langchain
def load_documents(dir):        
    loader = DirectoryLoader(dir, glob="*.txt", show_progress=True) # show progress bar
    docs = loader.load()
    return docs

# split documents into chunks
def split_documents(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

# embed the document chunks and store in chroma
def embed_docs(docs):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever