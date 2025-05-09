from Bio import Entrez, Medline
import chromadb
from sentence_transformers import SentenceTransformer
import time

# Configure Entrez
Entrez.email = "ankushchhabra051@gmail.com"  # Replace with your email
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")  # Using persistent storage
collection = client.get_or_create_collection(name="pubmed-central")

def fetch_pubmed_data(search_term, max_results=100):
    print(f"\n=== Fetching from PubMed ===")
    print(f"Search term: {search_term}")
    print(f"Max results requested: {max_results}")
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    print(f"Found {len(record['IdList'])} articles in PubMed")
    
    # Fetch details for each article
    id_list = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    
    documents = []
    for record in records:
        if 'AB' in record:  # AB is the abstract field
            # Combine title and abstract
            text = f"Title: {record.get('TI', 'No Title')}\nAbstract: {record['AB']}"
            pmid = record.get('PMID', '')
            print(f"Processing PMID {pmid}: {record.get('TI', 'No Title')[:100]}...")
            documents.append({
                'id': pmid,
                'text': text
            })
    
    print(f"\nSuccessfully processed {len(documents)} documents with abstracts")
    return documents

def add_to_chroma(documents):
    print(f"\nAdding {len(documents)} documents to ChromaDB")
    
    batch_size = 50
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc['text'] for doc in batch]
        
        # Generate and print embeddings info
        embeddings = model.encode(texts)
        print(f"\nBatch {i//batch_size + 1} embedding stats:")
        print(f"Shape: {embeddings.shape}")
        print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")
        
        try:
            collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=[doc['id'] for doc in batch]
            )
            total_added += len(batch)
            print(f"Added batch {i//batch_size + 1} ({len(batch)} documents)")
            print(f"Total documents added so far: {total_added}")
        except Exception as e:
            print(f"Error adding batch: {e}")
        
        time.sleep(1)  # Prevent overwhelming the system

def main():
    # More specific medical search terms
    search_terms = [
        "diabetes type 2 treatment guidelines",
        "hypertension management recent advances",
        "covid 19 treatment protocols",
        "alzheimer's disease current research",
        "obesity management strategies",
        "asthma treatment guidelines",
        "heart failure management",
        "cancer immunotherapy advances"
    ]
    
    for term in search_terms:
        print(f"\n=== Processing search term: {term} ===")
        documents = fetch_pubmed_data(term)
        print(f"Found {len(documents)} documents")
        add_to_chroma(documents)
        print(f"Completed adding documents for: {term}")
        
    # Print final collection stats
    print("\n=== Final Collection Statistics ===")
    print(f"Total documents in ChromaDB: {collection.count()}")

if __name__ == "__main__":
    main()
