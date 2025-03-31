import os
import gc
import torch
import logging
from pathlib import Path
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_document():
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = data_dir / "medical_test.txt"
    test_content = """
    Medical Information Document

    1. HIV and AIDS:
    HIV (Human Immunodeficiency Virus) attacks the body's immune system. The virus can be transmitted through blood, sexual contact, and from mother to child. AIDS (Acquired Immune Deficiency Syndrome) is the most severe phase of HIV infection.

    2. Diabetes:
    Diabetes is a metabolic disease that causes high blood sugar. The main types are Type 1 and Type 2 diabetes. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.

    3. Cancer:
    Cancer is a disease where cells grow uncontrollably and spread to other parts of the body. Treatment options include surgery, chemotherapy, and radiation therapy.

    4. Heart Disease:
    Cardiovascular disease affects the heart and blood vessels. Risk factors include high blood pressure, smoking, and high cholesterol.
    """
    test_file.write_text(test_content)
    print("Created test document")
    return test_file

def create_vector_db():
    try:
        # Force CPU usage and limit memory
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        torch.set_num_threads(1)
        
        # Initialize embeddings (simple approach)
        embeddings = SentenceTransformerEmbeddings(
            model_name=settings.EMBEDDINGS,
            model_kwargs={'device': 'cpu'}
        )
        print("Embeddings initialized")

        # Create test document if no documents exist
        data_dir = Path(settings.DATA_DIR)
        if not data_dir.exists() or not any(data_dir.glob("*.txt")):
            test_file = create_test_document()
        
        # Load documents
        loader = DirectoryLoader(
            str(data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=70
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} text chunks")
        
        if texts:
            # Create vector store
            qdrant = Qdrant.from_documents(
                texts,
                embeddings,
                url=settings.VECTOR_DB_URL,
                prefer_grpc=False,
                collection_name=settings.VECTOR_DB_NAME
            )
            print("Vector DB Successfully Created!")
            return True
        else:
            print("No texts to process")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    print("Starting vector database creation...")
    success = create_vector_db()
    
    if success:
        print("Successfully completed vector database creation")
    else:
        print("Failed to create vector database")