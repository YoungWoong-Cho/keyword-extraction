import argparse
import logging
from pathlib import Path
import datetime
import time
import csv

from keyword_extractor import KeywordExtractor, KeywordMethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveDemo:
    def __init__(self):
        """Initialize the interactive demo"""
        self.data_dir, self.state_dir = self._create_demo_directories()
        
        self.extraction_method = KeywordMethod.CUSTOM
        self.max_keywords = 100
        self.qdrant_location = ":memory:" # in-memory
        self.collection_name = "demo_documents"
        
        self.extractor = None
        self.reset_extractor()
        
        self.processed_docs = set()
        
        self.documents = {}
    
    def _create_demo_directories(self):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        state_dir = Path("state")
        state_dir.mkdir(exist_ok=True)
        
        return data_dir, state_dir
    
    def reset_extractor(self):
        self.extractor = KeywordExtractor(
            method=self.extraction_method,
            max_keywords=self.max_keywords,
            qdrant_location=self.qdrant_location,
            qdrant_collection_name=self.collection_name
        )
        
        self.processed_docs = set()
    
    def change_extraction_method(self, method_name: str):
        """Change the keyword extraction method"""
        method_map = {
            "custom": KeywordMethod.CUSTOM,
            "tfidf": KeywordMethod.TFIDF,
            "textrank": KeywordMethod.TEXTRANK,
            "rake": KeywordMethod.RAKE,
            "yake": KeywordMethod.YAKE,
            "kpminer": KeywordMethod.KPMINER
        }
        
        if method_name.lower() not in method_map:
            print(f"Unknown method: {method_name}")
            print(f"Available methods: {', '.join(method_map.keys())}")
            return
        
        self.extraction_method = method_map[method_name.lower()]
        
        self.reset_extractor()
        self.processed_docs = set()
        
        print(f"Changed extraction method to: {method_name}")
        print("Previous keywords have been cleared. Please reprocess your documents.")
    
    def load_warning_letters(self, csv_file_path):
        csv_path = Path(csv_file_path)
        if not csv_path.exists():
            print(f"File not found: {csv_file_path}")
            return False
        
        try:
            print(f"Loading warning letters from {csv_file_path}...")
            document_count = 0
            self.documents = {}
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    company = row.get('company_name', '').strip()
                    date = row.get('letter_issue_date', '').strip()
                    content = row.get('warning_letter_content', '').strip()
                    
                    if not content:
                        continue
                        
                    doc_id = f"letter_{i+1}"
                    if company and date:
                        doc_id = f"{company}_{date}".replace(' ', '_')
                    
                    self.documents[doc_id] = content
                    document_count += 1
            
            print(f"Successfully loaded {document_count} warning letters!")
            return True
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
    
    def process_sample_documents(self):
        """Process the sample documents"""
        if not self.documents:
            print("No documents loaded. Please load warning letters first.")
            return None
            
        print(f"Processing {len(self.documents)} documents...")
        start_time = time.time()
        
        if self.extraction_method != self.extractor.method:
            print(f"Extraction method changed to {self.extraction_method.value}. Resetting extractor...")
            self.reset_extractor()
            self.processed_docs = set()
        
        results = self.extractor.batch_process_documents(self.documents)
        self.processed_docs.update(self.documents.keys())
        
        duration = time.time() - start_time
        print(f"Processing completed in {duration:.2f} seconds")
        
        return results
    
    def process_custom_document(self, doc_id: str, text: str):
        print(f"Processing document: {doc_id}")
        start_time = time.time()
        
        keywords = self.extractor.process_document(doc_id, text)
        self.processed_docs.add(doc_id)
        
        duration = time.time() - start_time
        print(f"Processing completed in {duration:.2f} seconds")
        
        return keywords
    
    def compare_methods(self, text: str, doc_id: str = "comparison_doc"):
        print("\n" + "="*50)
        print("KEYWORD EXTRACTION METHOD COMPARISON")
        print("="*50)
        
        original_method = self.extraction_method
        results = {}
        
        for method in KeywordMethod:
            self.extractor.set_extraction_method(method)
            print(f"\nUsing method: {method.value}")
            
            start_time = time.time()
            keywords = self.extractor.extract_keywords(text, top_n=10)
            duration = time.time() - start_time
            
            results[method.value] = {
                "keywords": keywords,
                "time": duration
            }
            
            print(f"Extraction time: {duration:.4f} seconds")
            print("Top 10 keywords:")
            for idx, keyword in enumerate(keywords, 1):
                print(f"  {idx}. {keyword}")
        
        self.extractor.set_extraction_method(original_method)
        return results
    
    def show_top_keywords(self, n: int = 10):
        top_keywords = self.extractor.get_top_keywords(n)
        
        print("\nTop keywords across all documents:")
        print("-"*40)
        for idx, (keyword, count) in enumerate(top_keywords, 1):
            print(f"{idx}. {keyword}: {count}")
    
    def search_similar_documents(self, query: str, limit: int = 3):
        print(f"\nSearching for documents similar to: '{query}'")
        print("-"*40)
        
        similar_docs = self.extractor.find_similar_documents(query, limit=limit)
        
        if not similar_docs:
            print("No similar documents found.")
            return
        
        for idx, doc in enumerate(similar_docs, 1):
            print(f"Result {idx}:")
            print(f"  Document: {doc['id']}")
            print(f"  Similarity: {doc['score']:.4f}")
            print(f"  Keywords: {', '.join(doc['keywords'][:5])}")
            print(f"  Preview: {doc['text_preview'][:100]}...")
            print()
    
    def get_documents_with_keyword(self, keyword: str, limit: int = 5):
        print(f"\nDocuments containing keyword: '{keyword}'")
        print("-"*40)
        
        doc_ids = self.extractor.get_documents_with_keyword(keyword, limit)
        
        if not doc_ids:
            print(f"No documents found with keyword: {keyword}")
            return
        
        for idx, doc_id in enumerate(doc_ids, 1):
            print(f"{idx}. {doc_id}")
    
    def reprocess_documents(self):
        print("\nReprocessing all documents with current extraction method...")
        start_time = time.time()
        
        results = self.extractor.reprocess_all_documents()
        
        duration = time.time() - start_time
        print(f"Reprocessing completed in {duration:.2f} seconds")
        print(f"Updated {len(results)} documents")
        
        return results
    
    def save_current_state(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.state_dir / f"extractor_state_{timestamp}.pkl"
        
        self.extractor.save_state(filepath)
        print(f"State saved to: {filepath}")
        
        return filepath
    
    def load_state(self, filepath: str):
        try:
            self.extractor.load_state(filepath)
            self.extraction_method = self.extractor.method
            print(f"State loaded from: {filepath}")
            
            self.processed_docs = set(self.extractor.document_keywords.keys())
            
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def show_document_keywords(self, doc_id: str, n: int = 10):
        if not self.processed_docs:
            print("No documents have been processed yet.")
            return
            
        if doc_id not in self.processed_docs:
            print(f"Document ID '{doc_id}' not found in processed documents.")
            print("Available document IDs:")
            for i, doc in enumerate(sorted(self.processed_docs), 1):
                print(f"  {i}. {doc}")
            return
            
        keywords = self.extractor.get_document_keywords(doc_id, top_n=n)
        
        print(f"\nTop {len(keywords)} keywords for document: {doc_id}")
        print("-"*40)
        for idx, keyword in enumerate(keywords, 1):
            print(f"{idx}. {keyword}")
        
        if doc_id in self.documents:
            preview = self.documents[doc_id][:150] + "..." if len(self.documents[doc_id]) > 150 else self.documents[doc_id]
            print(f"\nDocument preview: \n{preview}")

    def menu(self):
        while True:
            print("\n" + "="*50)
            print("KEYWORD EXTRACTION INTERACTIVE DEMO")
            print("="*50)
            print(f"Current method: {self.extraction_method.value}")
            print(f"Documents processed: {len(self.processed_docs)}")
            print(f"Total keywords: {len(self.extractor.global_keywords)}")
            print(f"Documents loaded: {len(self.documents)}")
            print("="*50)
            print("1. Load warning letters from CSV")
            print("2. Process loaded documents")
            print("3. Process custom document")
            print("4. Change extraction method")
            print("5. Show keywords from specific document")
            print("6. Show top keywords")
            print("7. Find documents with keyword")
            print("8. Save current state")
            print("9. Load saved state")
            print("0. Exit")
            print("="*50)
            
            choice = input("Enter your choice: ")
            
            if choice == "1":
                csv_path = input("Enter the path to warning_letter.csv: ") or "warning_letter.csv"
                self.load_warning_letters(csv_path)
                
            elif choice == "2":
                self.process_sample_documents()
                self.show_top_keywords()
                
            elif choice == "3":
                doc_id = input("Enter document ID: ")
                print("Enter document text (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                
                text = "\n".join(lines)
                if text:
                    keywords = self.process_custom_document(doc_id, text)
                    print(f"Extracted keywords: {', '.join(keywords)}")
                
            elif choice == "4":
                methods = [m.value for m in KeywordMethod]
                print(f"Available methods: {', '.join(methods)}")
                method = input("Enter method name: ")
                self.change_extraction_method(method)
                
            elif choice == "5":
                if not self.processed_docs:
                    print("No documents have been processed yet.")
                else:
                    # Show a count if there are too many documents
                    if len(self.processed_docs) > 20:
                        print(f"There are {len(self.processed_docs)} processed documents.")
                        doc_id = input("Enter document ID: ")
                    else:
                        print("Available document IDs:")
                        for i, doc in enumerate(sorted(self.processed_docs), 1):
                            print(f"  {i}. {doc}")
                        doc_id = input("Enter document ID: ")
                    
                    try:
                        n = int(input("How many keywords to show? [10]: ") or "10")
                        self.show_document_keywords(doc_id, n)
                    except ValueError:
                        print("Please enter a valid number")
                
            elif choice == "6":
                try:
                    n = int(input("How many keywords to show? [10]: ") or "10")
                    self.show_top_keywords(n)
                except ValueError:
                    print("Please enter a valid number")
                
            elif choice == "7":
                keyword = input("Enter keyword to search for: ")
                try:
                    limit = int(input("Number of documents to show? [5]: ") or "5")
                    self.get_documents_with_keyword(keyword, limit)
                except ValueError:
                    print("Please enter a valid number")
                
            elif choice == "8":
                filepath = self.save_current_state()
                
            elif choice == "9":
                filepath = input("Enter state file path: ")
                self.load_state(filepath)
                
            elif choice == "0":
                print("Exiting demo...")
                break
                
            else:
                print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Keyword Extraction Interactive Demo")
    parser.add_argument("--method", type=str, default="custom", 
                      help="Initial extraction method (custom, tfidf, textrank, rake, yake, kpminer)")
    args = parser.parse_args()
    
    demo = InteractiveDemo()
    
    if args.method:
        demo.change_extraction_method(args.method)
    
    demo.menu()

if __name__ == "__main__":
    main() 
