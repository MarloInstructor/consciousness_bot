from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("./content/data").load_data()

text_splitter = SentenceSplitter(chunk_size=800, chunk_overlap=150)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "./content/embedding_model/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model, cache_folder=embeddings_folder
)

vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[text_splitter], embed_model=embeddings
)

vector_index.storage_context.persist(persist_dir="./content/vector_index")