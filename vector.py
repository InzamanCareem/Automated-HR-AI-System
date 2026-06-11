from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


class LTM:
    def __init__(self, path="ltm_store"):
        self.path = path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        self.store = self._load() if Path(path).exists() else FAISS.from_texts(["placeholder"], self.embeddings)

    def _load(self):
        print(f"[LTM-LOAD] Reading from disk: {self.path}")
        return FAISS.load_local(self.path, self.embeddings, allow_dangerous_deserialization=True)

    def store_memory(self, employee_id, text):
        print(f"[LTM-STORE] Adding: {text}")
        self.store.add_texts(texts=[text], metadatas=[{"employee_id": employee_id}])
        self.store.save_local(self.path)

    def retrieve_memory(self, employee_id, query, k=20):
        print(f"[LTM-SEARCH] Query: {query}")
        results = self.store.similarity_search_with_score(query, k=k)
        memories = []

        for doc, score in results:
            if score < 0.75:
                memories.append(doc.page_content)

        for i, mem in enumerate(memories, 1):
            print(f"Match {i}: {mem}")

        return memories
