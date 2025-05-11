from Bio import Entrez, Medline
import chromadb
from sentence_transformers import SentenceTransformer
import time
from typing import List

# Configure Entrez
Entrez.email = "ankushchhabra051@gmail.com"  # Replace with your email
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")  # Using persistent storage
collection = client.get_or_create_collection(name="pubmed-central")

def create_chunks(text: str, chunk_size: int = 150, overlap: int = 15) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    print(f"\n=== Chunking Text ===")
    print(f"Total words: {len(words)}")
    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap size: {overlap} words")
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= overlap:
            chunk_text = " ".join(chunk)
            chunks.append(chunk_text)
            print(f"\nChunk {len(chunks)}:")
            print(f"Start word: {i}")
            print(f"End word: {i + len(chunk)}")
            print(f"Chunk length: {len(chunk)} words")
            print(f"Preview: {chunk_text[:100]}...")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    return chunks

def fetch_pubmed_data(search_term, category, subcategory=None, max_results=100):
    print(f"\n=== Fetching from PubMed ===")
    print(f"Category: {category}")
    print(f"Subcategory: {subcategory}")
    print(f"Search term: {search_term}")
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    num_articles = len(record['IdList'])
    print(f"Found {num_articles} articles in PubMed")
    
    # Fetch details for each article
    id_list = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = Medline.parse(handle)
    
    documents = []
    total_chunks = 0
    articles_with_abstract = 0
    
    for record in records:
        if 'AB' in record:
            articles_with_abstract += 1
            print(f"\n=== Processing Article ===")
            print(f"Title: {record.get('TI', 'No Title')[:100]}...")
            print(f"PMID: {record.get('PMID', '')}")
            
            text = f"Title: {record.get('TI', 'No Title')}\nAbstract: {record['AB']}"
            pmid = record.get('PMID', '')
            
            print(f"\nOriginal text length: {len(text.split())} words")
            chunks = create_chunks(text)
            total_chunks += len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{pmid}_{i}"
                metadata = {
                    'category': category,
                    'subcategory': subcategory,
                    'source': 'pubmed',
                    'pmid': pmid,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'title': record.get('TI', 'No Title'),
                    'authors': '; '.join(record.get('AU', [])),  # Convert list to string
                    'publication_date': record.get('DP', ''),
                    'journal': record.get('JT', ''),
                    'mesh_terms': '; '.join(record.get('MH', [])),  # Convert list to string
                    'search_term': search_term
                }
                
                documents.append({
                    'id': chunk_id,
                    'text': chunk,
                    'metadata': metadata
                })
    
    print(f"\n=== Chunking Summary ===")
    print(f"Total articles processed: {num_articles}")
    print(f"Articles with abstracts: {articles_with_abstract}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per article: {total_chunks / articles_with_abstract if articles_with_abstract > 0 else 0:.2f}")
    
    print(f"\nSuccessfully processed {len(documents)} documents with abstracts")
    return documents

def add_to_chroma(documents):
    print(f"\n=== Adding Documents to ChromaDB ===")
    print(f"Total chunks to add: {len(documents)}")
    
    batch_size = 50
    total_added = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc['text'] for doc in batch]
        
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        print(f"Batch size: {len(batch)} chunks")
        print(f"Sample chunk preview: {texts[0][:100]}...")
        
        embeddings = model.encode(texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        try:
            collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=[doc['id'] for doc in batch],
                metadatas=[doc['metadata'] for doc in batch]
            )
            total_added += len(batch)
            print(f"Successfully added batch {i//batch_size + 1}")
            print(f"Progress: {total_added}/{len(documents)} chunks ({(total_added/len(documents)*100):.1f}%)")
            
            # Print some metadata from the batch
            print("\nSample metadata from batch:")
            for idx, meta in enumerate(batch[:2]):
                print(f"Chunk {idx+1}:")
                print(f"- PMID: {meta['metadata']['pmid']}")
                print(f"- Chunk: {meta['metadata']['chunk_id']}/{meta['metadata']['total_chunks']}")
                print(f"- Category: {meta['metadata']['category']}/{meta['metadata']['subcategory']}")
            
        except Exception as e:
            print(f"Error adding batch: {e}")
        
        time.sleep(1)
    
    print(f"\n=== ChromaDB Upload Complete ===")
    print(f"Total chunks added: {total_added}")
    print(f"Final collection size: {collection.count()}")

def search_documents(query_text, category=None, limit=3):
    """
    Search documents using both semantic similarity and metadata filtering
    """
    print(f"\n=== Searching documents ===")
    print(f"Query: {query_text}")
    print(f"Category filter: {category}")
    
    try:
        # Generate embedding for query
        query_embedding = model.encode(query_text).tolist()
        
        # Build where clause
        where = {}
        if category:
            where["category"] = category
        
        # Query collection with detailed logging
        print("Executing ChromaDB query...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["metadatas", "distances", "documents"]
        )
        
        if not results["ids"]:
            print("No matching documents found")
            return []
            
        print(f"Found {len(results['documents'][0])} matching documents")
        
        # Format results with metadata and scores
        formatted_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - dist  # Convert distance to similarity score
            formatted_results.append({
                "document": doc,
                "metadata": meta,
                "similarity": similarity,
                "rank": i + 1
            })
            
            # Print detailed match information
            print(f"\nMatch {i+1} (similarity: {similarity:.4f})")
            print(f"Category: {meta.get('category')}/{meta.get('subcategory')}")
            print(f"Title: {meta.get('title')[:100]}...")
            
        return formatted_results
            
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def get_collection_stats():
    """Get statistics about the document collection"""
    try:
        doc_count = collection.count()
        print(f"\nTotal documents: {doc_count}")
        
        # Get sample to analyze categories
        if doc_count > 0:
            results = collection.query(
                query_embeddings=[model.encode("medical").tolist()],
                n_results=doc_count,
                include=["metadatas"]
            )
            
            categories = {}
            for meta in results["metadatas"][0]:
                cat = meta.get("category", "unknown")
                subcat = meta.get("subcategory", "unknown")
                if cat not in categories:
                    categories[cat] = {}
                if subcat not in categories[cat]:
                    categories[cat][subcat] = 0
                categories[cat][subcat] += 1
            
            print("\nDocument distribution:")
            for cat, subcats in categories.items():
                print(f"\n{cat}:")
                for subcat, count in subcats.items():
                    print(f"  {subcat}: {count} documents")
                    
    except Exception as e:
        print(f"Error getting stats: {e}")

def main():
    # Organized medical topics with categories
    medical_topics = {
        "endocrinology": {
            "diabetes": [
                "diabetes type 2 treatment guidelines",
                "diabetes management protocol",
                "diabetic complications prevention"
            ],
            "obesity": [
                "obesity management strategies",
                "weight loss interventions"
            ]
        },
        "cardiology": {
            "heart_failure": [
                "heart failure management",
                "cardiac treatment advances"
            ],
            "hypertension": [
                "hypertension management recent advances",
                "blood pressure control guidelines"
            ]
        },
        "respiratory": {
            "asthma": [
                "asthma treatment guidelines",
                "bronchial asthma therapy"
            ]
        },
        "neurology": {
            "alzheimer": [
                "alzheimer's disease current research",
                "dementia treatment advances"
            ]
        },
        "infectious_disease": {
            "covid19": [
                "covid 19 treatment protocols",
                "coronavirus management guidelines"
            ]
        },
        "oncology": {
            "immunotherapy": [
                "cancer immunotherapy advances",
                "tumor treatment innovations"
            ]
        }
    }
    
    for category, subcategories in medical_topics.items():
        for subcategory, terms in subcategories.items():
            for term in terms:
                print(f"\n=== Processing: {category}/{subcategory} - {term} ===")
                documents = fetch_pubmed_data(term, category, subcategory)
                add_to_chroma(documents)
                print(f"Completed adding documents for {category}/{subcategory}")

    # Print final collection stats
    print("\n=== Final Collection Statistics ===")
    print(f"Total documents in ChromaDB: {collection.count()}")
    
    # Print collection statistics after adding documents
    print("\n=== Collection Statistics ===")
    get_collection_stats()

if __name__ == "__main__":
    main()