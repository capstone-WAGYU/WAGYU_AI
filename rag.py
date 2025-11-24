from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


from bs4 import BeautifulSoup
import requests, os
from dotenv import load_dotenv
import tavily

load_dotenv()
tavily_client = tavily.TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ---------------------- URL 크롤러 ----------------------
def crawl(url):
    try:
        html = requests.get(url, timeout=7, headers={"User-Agent":"Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(ps)
    except:
        return ""

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

q = input()

search = tavily_client.search(q, max_results=5)
urls = [r["url"] for r in search["results"]]

crawl_data = [crawl(u) for u in urls]
docs = splitter.create_documents(crawl_data)
vectordb = FAISS.from_documents(docs, emb)
retriever = vectordb.as_retriever()
#chat(retriever=retriver) 쓰면 될 듯
