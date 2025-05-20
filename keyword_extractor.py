from typing import List, Dict, Tuple
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import pickle
import logging
from tqdm import tqdm
from enum import Enum
import re
import string
import math
import uuid
import hashlib
import ssl
import sys

# For vector database
import qdrant_client
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordMethod(Enum):
    CUSTOM = "custom"
    TFIDF = "tfidf"
    TEXTRANK = "textrank"
    RAKE = "rake"
    YAKE = "yake"
    KPMINER = "kpminer"

class KeywordExtractor:
    def __init__(self, 
                 method: KeywordMethod = KeywordMethod.CUSTOM,
                 model_name: str = "all-MiniLM-L6-v2", 
                 max_keywords: int = 1000,
                 min_keyword_freq: int = 2,
                 qdrant_location: str = ":memory:",
                 qdrant_collection_name: str = "documents"):
        """
        Initialize the KeywordExtractor with specified parameters.
        
        Args:
            method: The keyword extraction method to use
            model_name: The sentence transformer model to use for embeddings
            max_keywords: Maximum number of keywords to maintain in the global set
            min_keyword_freq: Minimum frequency for a term to be considered a keyword
            qdrant_location: Location of Qdrant database (use :memory: for in-memory)
            qdrant_collection_name: Name of the Qdrant collection
        """
        # Fix SSL certificate issues (common on macOS)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            
        # Download necessary NLTK data
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('stopwords')
                nltk.download('punkt')
                nltk.download('punkt_tab')
            except Exception as e:
                logger.error(f"Error downloading NLTK data: {e}")
                logger.info("You may need to manually install NLTK data. See https://www.nltk.org/data.html")
            
        # Initialize NLP components
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found. Downloading it now...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available. Using a minimal set of stop words.")
            self.stop_words = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                              "when", "where", "how", "who", "which", "this", "that", "to", "of", 
                              "in", "for", "with", "on", "by", "at", "from"}
        
        # Set extraction method
        self.method = method
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize vector DB
        self.qdrant_client = qdrant_client.QdrantClient(location=qdrant_location)
        self._initialize_collection(qdrant_collection_name)
        self.collection_name = qdrant_collection_name
        
        # Keywords management
        self.max_keywords = max_keywords
        self.min_keyword_freq = min_keyword_freq
        self.global_keywords = {}  # Track keyword frequencies
        self.document_keywords = {}  # Map document IDs to their keywords
        self.keyword_to_docs = {}  # Map keywords to documents containing them
        
        # TF-IDF for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        self.tfidf_fitted = False
        self.document_corpus = {}  # Store document texts for TF-IDF training
        
        # KP-Miner corpus statistics
        self.term_document_freq = {}  # Number of documents containing each term
        self.corpus_size = 0  # Number of documents in corpus
        
        # Initialize TextRank components
        self.window_size = 4
        self.edge_weight = 1.0
        
        # Initialize RAKE components
        self.rake_min_chars = 3
        self.rake_max_words = 3
        
        # Initialize YAKE components
        self.yake_max_ngram_size = 3
        self.yake_deduplication_threshold = 0.9
        
        # Initialize KP Miner components
        self.kpminer_min_df = 2
        self.kpminer_max_df = 0.8
        
    def _initialize_collection(self, collection_name: str):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Error initializing Qdrant collection: {e}")
            logger.info("Creating a new collection anyway")
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
            except Exception as e2:
                logger.error(f"Failed to create collection: {e2}")
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from a single document using the selected method.
        
        Args:
            text: The document text
            top_n: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Handle empty or very short texts
        if not text or len(text.strip()) < 3:
            return []
            
        try:
            if self.method == KeywordMethod.CUSTOM:
                return self._extract_keywords_custom(text, top_n)
            elif self.method == KeywordMethod.TFIDF:
                return self._extract_keywords_tfidf(text, top_n)
            elif self.method == KeywordMethod.TEXTRANK:
                return self._extract_keywords_textrank(text, top_n)
            elif self.method == KeywordMethod.RAKE:
                return self._extract_keywords_rake(text, top_n)
            elif self.method == KeywordMethod.YAKE:
                return self._extract_keywords_yake(text, top_n)
            elif self.method == KeywordMethod.KPMINER:
                return self._extract_keywords_kpminer(text, top_n)
            else:
                logger.warning(f"Unknown method: {self.method}. Using custom method.")
                return self._extract_keywords_custom(text, top_n)
        except Exception as e:
            logger.error(f"Error extracting keywords with method {self.method}: {e}")
            # Fall back to custom method if any other method fails
            try:
                return self._extract_keywords_custom(text, top_n)
            except Exception as e2:
                logger.error(f"Fallback extraction also failed: {e2}")
                return []
    
    def _extract_keywords_custom(self, text: str, top_n: int = 10) -> List[str]:
        """Custom keyword extraction method using spaCy"""
        # Preprocess text
        doc = self.nlp(text.lower())
        
        # Filter tokens - keep only nouns, verbs, adjectives
        filtered_tokens = [token.text for token in doc if 
                          token.pos_ in {'NOUN', 'PROPN', 'ADJ', 'VERB'} and 
                          token.text not in self.stop_words and
                          len(token.text) > 2]
        
        # Get noun phrases
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                       if len(chunk.text) > 3 and chunk.text.lower() not in self.stop_words]
        
        # Combine tokens and noun phrases for candidate keywords
        candidates = filtered_tokens + noun_phrases
        
        if not candidates:
            return []
            
        # Count frequencies for ranking
        keyword_counts = {}
        for kw in candidates:
            if kw in keyword_counts:
                keyword_counts[kw] += 1
            else:
                keyword_counts[kw] = 1
        
        # Get top keywords by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [kw for kw, _ in sorted_keywords[:top_n]]
        
        return keywords
    
    def _extract_keywords_tfidf(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using TF-IDF approach"""        
        try:
            # If we don't have a fitted vectorizer yet, use custom method as fallback
            if not self.tfidf_fitted or len(self.document_corpus) < 2:
                logger.warning("TF-IDF vectorizer not fitted with multiple documents yet. Falling back to custom method.")
                return self._extract_keywords_custom(text, top_n)
            
            # Transform the new document
            text_vector = self.tfidf_vectorizer.transform([text])
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get scores for the document
            scores = zip(feature_names, text_vector.toarray()[0])
            
            # Sort by score
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Get top keywords
            keywords = [word for word, score in sorted_scores[:top_n]]
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Error in TF-IDF extraction: {e}. Falling back to basic extraction.")
            return self._extract_keywords_custom(text, top_n)
    
    def _extract_keywords_textrank(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using TextRank algorithm"""
        # For very short texts, fall back to custom method
        if len(text.split()) < 5:
            return self._extract_keywords_custom(text, top_n)
            
        try:
            # Preprocess text
            doc = self.nlp(text.lower())
            
            # Keep only nouns and adjectives as nodes
            nodes = [token.text for token in doc if 
                    token.pos_ in {'NOUN', 'PROPN', 'ADJ'} and 
                    token.text not in self.stop_words and
                    len(token.text) > 2]
            
            # If not enough nodes, fall back to custom method
            if len(nodes) < 3:
                return self._extract_keywords_custom(text, top_n)
                
            # Build graph: add edges between words that co-occur in a window
            graph = {}
            for i, node in enumerate(nodes):
                if node not in graph:
                    graph[node] = {}
                    
                # Add edges with words in the window
                window_start = max(0, i - self.window_size)
                window_end = min(len(nodes), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i != j:
                        neighbor = nodes[j]
                        if neighbor not in graph:
                            graph[neighbor] = {}
                        
                        # Add or update edge weight
                        if neighbor in graph[node]:
                            graph[node][neighbor] += self.edge_weight
                        else:
                            graph[node][neighbor] = self.edge_weight
                            
                        if node in graph[neighbor]:
                            graph[neighbor][node] += self.edge_weight
                        else:
                            graph[neighbor][node] = self.edge_weight
            
            # Run TextRank algorithm (simplified version)
            # Initialize scores
            scores = {node: 1.0 for node in graph}
            
            # Iterative update
            max_iterations = 50
            damping = 0.85
            threshold = 0.0001
            
            for _ in range(max_iterations):
                prev_scores = scores.copy()
                
                # Update score for each node
                for node in graph:
                    score_sum = 0
                    # Skip isolated nodes (no neighbors)
                    if not graph[node]:
                        continue
                        
                    for neighbor, weight in graph[node].items():
                        # Skip neighbors with no connections
                        if not graph[neighbor]:
                            continue
                        score_sum += weight * prev_scores[neighbor] / sum(graph[neighbor].values())
                    
                    scores[node] = (1 - damping) + damping * score_sum
                
                # Check convergence
                if all(abs(scores[node] - prev_scores[node]) < threshold for node in scores):
                    break
            
            # Sort by score and get top keywords
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, score in sorted_scores[:top_n]]
            
            return keywords
        except Exception as e:
            logger.warning(f"Error in TextRank extraction: {e}. Falling back to basic extraction.")
            return self._extract_keywords_custom(text, top_n)
    
    def _extract_keywords_rake(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using RAKE (Rapid Automatic Keyword Extraction)"""
        # For very short texts, fall back to custom method
        if len(text.split()) < 5:
            return self._extract_keywords_custom(text, top_n)
            
        try:
            # Clean text - remove punctuation and convert to lowercase
            text = text.lower()
            
            # Replace punctuation with spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Split text by stop words to get candidate phrases
            # First, create a regex pattern of stop words
            stop_words_pattern = r'\b(' + r'|'.join(self.stop_words) + r')\b'
            
            # Split the text by stop words or punctuation
            # This gives us phrases where stop words act as delimiters
            candidate_chunks = re.split(stop_words_pattern, text)
            
            # Clean the chunks and remove any that are empty or too short
            phrases = []
            for chunk in candidate_chunks:
                # Clean extra whitespace
                chunk = re.sub(r'\s+', ' ', chunk).strip()
                
                # Add valid phrases
                if chunk and len(chunk) >= self.rake_min_chars:
                    phrases.append(chunk)
            
            # If not enough phrases, fall back to custom method
            if len(phrases) < 2:
                return self._extract_keywords_custom(text, top_n)
            
            # Filter out short phrases or phrases with too many words
            phrases = [
                phrase for phrase in phrases 
                if len(phrase) >= self.rake_min_chars and 
                len(phrase.split()) <= self.rake_max_words
            ]
            
            # If still not enough phrases, fall back to custom method
            if not phrases:
                return self._extract_keywords_custom(text, top_n)
            
            # Calculate word frequency and degree
            word_freq = {}
            word_degree = {}
            
            for phrase in phrases:
                words = phrase.split()
                degree = len(words) - 1
                
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                    word_degree[word] = word_degree.get(word, 0) + degree
            
            # Calculate word scores
            word_scores = {}
            for word in word_freq:
                word_scores[word] = word_degree[word] / max(1, word_freq[word])
            
            # Calculate phrase scores
            phrase_scores = {}
            for phrase in phrases:
                words = phrase.split()
                score = sum(word_scores.get(word, 0) for word in words)
                phrase_scores[phrase] = score
            
            # Sort phrases by score
            sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top keywords
            keywords = [phrase for phrase, score in sorted_phrases[:top_n]]
            
            return keywords
        except Exception as e:
            logger.warning(f"Error in RAKE extraction: {e}. Falling back to basic extraction.")
            return self._extract_keywords_custom(text, top_n)
    
    def _extract_keywords_yake(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using YAKE (Yet Another Keyword Extractor)"""
        # For very short texts, fall back to custom method
        if len(text.split()) < 5:
            return self._extract_keywords_custom(text, top_n)
            
        try:
            # Preprocessing
            text = text.lower()
            sentences = nltk.sent_tokenize(text)
            
            # If not enough sentences, fall back to custom method
            if len(sentences) < 1:
                return self._extract_keywords_custom(text, top_n)
            
            # Extract n-grams
            all_candidates = []
            
            for n in range(1, self.yake_max_ngram_size + 1):
                for sentence in sentences:
                    words = [word for word in nltk.word_tokenize(sentence) 
                             if word not in self.stop_words and word not in string.punctuation]
                    
                    if len(words) >= n:
                        for i in range(len(words) - n + 1):
                            candidate = ' '.join(words[i:i+n])
                            all_candidates.append(candidate)
            
            # If not enough candidates, fall back to custom method
            if len(all_candidates) < 2:
                return self._extract_keywords_custom(text, top_n)
            
            # Calculate term frequency
            tf = {}
            for candidate in all_candidates:
                if candidate in tf:
                    tf[candidate] += 1
                else:
                    tf[candidate] = 1
            
            # Calculate terms in context
            candidates_context = {}
            for sentence in sentences:
                words = [word for word in nltk.word_tokenize(sentence)]
                
                for n in range(1, self.yake_max_ngram_size + 1):
                    if len(words) >= n:
                        for i in range(len(words) - n + 1):
                            candidate = ' '.join(words[i:i+n])
                            
                            if candidate in tf:
                                left_context = words[max(0, i-5):i]
                                right_context = words[i+n:min(len(words), i+n+5)]
                                context = left_context + right_context
                                
                                if candidate not in candidates_context:
                                    candidates_context[candidate] = set()
                                
                                candidates_context[candidate].update(context)
            
            # Calculate scores - lower is better in YAKE
            scores = {}
            for candidate, frequency in tf.items():
                # Term frequency component
                tf_component = frequency
                
                # Position component (not fully implemented, using a placeholder)
                position = 1.0
                
                # Context component
                context_size = len(candidates_context.get(candidate, set()))
                context_component = 1.0 / max(1, context_size)
                
                # Final score (lower is better in YAKE)
                scores[candidate] = context_component / (tf_component * position)
            
            # Sort by score (ascending since lower is better)
            sorted_candidates = sorted(scores.items(), key=lambda x: x[1])
            
            # Get top keywords
            keywords = [candidate for candidate, score in sorted_candidates[:top_n]]
            
            # Remove duplicate keywords (substrings)
            final_keywords = []
            for kw in keywords:
                if not any(kw in other and kw != other for other in final_keywords):
                    final_keywords.append(kw)
                    if len(final_keywords) >= top_n:
                        break
            
            return final_keywords
        except Exception as e:
            logger.warning(f"Error in YAKE extraction: {e}. Falling back to basic extraction.")
            return self._extract_keywords_custom(text, top_n)
    
    def _extract_keywords_kpminer(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords using KP-Miner algorithm as described in:
        El-Beltagy & Rafea (2010) "KP-Miner: Participation in SemEval-2"
        https://aclanthology.org/S10-1041.pdf
        """
        # For very short texts, fall back to custom method
        if len(text.split()) < 5:
            return self._extract_keywords_custom(text, top_n)
            
        try:
            # Preprocess text
            text = text.lower()
            
            # Count document length in words (for adaptive parameters)
            words = re.findall(r'\b\w+\b', text)
            doc_length = len(words)
            
            # Split into sentences for position analysis
            sentences = nltk.sent_tokenize(text)
            sentence_words = []
            for s in sentences:
                sentence_words.append([w for w in nltk.word_tokenize(s) 
                                     if w not in self.stop_words and w not in string.punctuation])
                
            if not sentences:
                return self._extract_keywords_custom(text, top_n)
            
            # 1. Generate candidate phrases using proper phrase splitting
            # Create a regex pattern for stop words and punctuation
            stop_pattern = r'\b(' + r'|'.join(self.stop_words) + r')\b|[' + string.punctuation + r']'
            
            # Split the text into candidate phrases
            candidate_chunks = re.split(stop_pattern, text)
            
            # Clean and filter the chunks
            candidates = []
            for chunk in candidate_chunks:
                if chunk and not chunk.isspace():
                    # Clean whitespace
                    chunk = re.sub(r'\s+', ' ', chunk).strip()
                    
                    # Add valid chunks (at least 3 chars, at most 5 words)
                    if len(chunk) >= 3 and len(chunk.split()) <= 5:
                        candidates.append(chunk)
            
            # 2. Calculate lasf and cutoff
            lasf = 3
            cutoff_words = 400
            
            # Count word positions to determine cutoff
            word_positions = {}
            word_count = 0
            
            for sentence in sentences:
                words_in_sentence = [w for w in nltk.word_tokenize(sentence) 
                                   if w not in self.stop_words and w not in string.punctuation]
                for word in words_in_sentence:
                    if word not in word_positions:
                        word_positions[word] = word_count
                    word_count += 1
            
            # If no valid candidates, fall back to custom method
            if not candidates:
                return self._extract_keywords_custom(text, top_n)
            
            # Calculate candidate frequencies
            candidate_freq = {}
            for candidate in candidates:
                if candidate in candidate_freq:
                    candidate_freq[candidate] += 1
                else:
                    candidate_freq[candidate] = 1
            
            # Filter by lasf (least allowable seen frequency)
            filtered_candidates = {k: v for k, v in candidate_freq.items() if v >= lasf}
            
            # If no candidates pass the filter, fall back to custom method
            if not filtered_candidates:
                return self._extract_keywords_custom(text, top_n)
            
            # Filter by position (must appear before cutoff)
            final_candidates = {}
            
            for candidate, freq in filtered_candidates.items():
                # Check first appearance position
                first_word = candidate.split()[0]
                
                # If the first word appears before cutoff, keep the candidate
                if first_word in word_positions and word_positions[first_word] <= cutoff_words:
                    final_candidates[candidate] = freq
            
            # If no candidates pass the filter, fall back to custom method
            if not final_candidates:
                return self._extract_keywords_custom(text, top_n)
            
            # 4. Calculate boosting factor exactly as in the paper:
            # B_i = |N_d|/(|P_d| * α), where:
            # |N_d| is the number of all candidate terms
            # |P_d| is the number of candidate phrases (multi-word)
            # α is a constant (set to 2.3 in the paper)
            # If B_i > σ then B_i = σ (σ is a constant, set to 3)
            
            # Count single-word and multi-word candidates
            all_terms_count = len(final_candidates)
            multi_word_terms_count = sum(1 for term in final_candidates if len(term.split()) > 1)
            
            # Avoid division by zero
            if multi_word_terms_count == 0:
                multi_word_terms_count = 1
                
            # Constants from the paper
            alpha = 2.3  # Value specified in the paper
            sigma = 3.0
            
            # Calculate boosting factor B_i
            boosting_factor = all_terms_count / (multi_word_terms_count * alpha)
            
            # Cap at sigma if needed
            if boosting_factor > sigma:
                boosting_factor = sigma
                
            # Position factor (set to 1 since we already filtered by position)
            position_factor = 1.0
            
            # 5. Calculate term weights according to the formula from the paper:
            # w_ij = tf_ij * idf * B_i * P_i
            # where:
            # - tf_ij = frequency of term j in Document D_i
            # - idf = log N; where N is the number of documents in the collection
            #   if the term is compound, n is set to 1
            # - B_i = boosting factor calculated above
            # - P_i = position factor (1.0)
            term_weights = {}
            
            # Calculate df = log N where N is corpus size
            # Use corpus statistics if available, otherwise set to 1
            corpus_size = max(1, self.corpus_size)  # Avoid division by zero
            
            # According to the formula in the paper, idf is log(N)
            idf_base = math.log2(corpus_size) if corpus_size > 1 else 1.0
            
            # If term is compound (multi-word), n is set to 1
            # Otherwise, if using position rules, P_i is set to 1
            
            for term, freq in final_candidates.items():
                # tf_ij is the frequency of the term in the document
                tf = freq
                
                # Apply the exact formula from the paper (multiply all factors)
                if len(term.split()) > 1:
                    # For multi-word terms, use boosting factor
                    # idf = log N as n=1 for compound terms
                    term_weight = tf * idf_base * boosting_factor * position_factor
                else:
                    # For single words, don't apply boosting (set boosting to 1.0)
                    # Try to get actual document frequency if available
                    doc_freq = self.term_document_freq.get(term, 1)
                    doc_freq = min(doc_freq, corpus_size)  # Can't exceed corpus size
                    idf = math.log2(corpus_size / doc_freq) if doc_freq > 0 else idf_base
                    term_weight = tf * idf * 1.0 * position_factor
                    
                term_weights[term] = term_weight
            
            # 6. Apply refinement step as described in the paper
            # Get the top candidates by weight
            temp_sorted_candidates = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
            top_phrases = [phrase for phrase, _ in temp_sorted_candidates[:top_n*2]]  # Get more than needed for refinement
            
            # Build a dictionary for phrase to weight mapping
            phrase_weights = {phrase: weight for phrase, weight in term_weights.items() if phrase in top_phrases}
            phrase_freqs = {phrase: final_candidates[phrase] for phrase in top_phrases}
            
            # Check for sub-phrases
            for phrase in top_phrases:
                words = phrase.split()
                # Skip single words - they can't have sub-phrases
                if len(words) < 2:
                    continue
                
                # Generate all possible sub-phrases and check if they exist in our top phrases
                for i in range(len(words)):
                    for j in range(i+1, len(words)+1):
                        sub_phrase = ' '.join(words[i:j])
                        if sub_phrase != phrase and sub_phrase in phrase_weights:
                            # Decrement the sub-phrase frequency by the frequency of the parent phrase
                            # Phrase "body weight" in "excess body weight" should be decremented
                            new_freq = max(1, phrase_freqs[sub_phrase] - phrase_freqs[phrase])
                            phrase_freqs[sub_phrase] = new_freq
                            
                            # Recalculate weight for the sub-phrase
                            if len(sub_phrase.split()) > 1:
                                phrase_weights[sub_phrase] = new_freq * idf_base * boosting_factor * position_factor
                            else:
                                doc_freq = self.term_document_freq.get(sub_phrase, 1)
                                doc_freq = min(doc_freq, corpus_size)
                                idf = math.log2(corpus_size / doc_freq) if doc_freq > 0 else idf_base
                                phrase_weights[sub_phrase] = new_freq * idf * 1.0 * position_factor
            
            # Rank candidates by updated weight
            sorted_candidates = sorted(phrase_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Get top keywords
            keywords = [candidate for candidate, score in sorted_candidates[:top_n]]
            
            return keywords
        except Exception as e:
            logger.warning(f"Error in KP-Miner extraction: {e}. Falling back to basic extraction.")
            return self._extract_keywords_custom(text, top_n)
    
    def process_document(self, doc_id: str, text: str) -> List[str]:
        """
        Process a document to extract keywords and update the vector DB.
        
        Args:
            doc_id: Unique document identifier
            text: Document content
            
        Returns:
            List of keywords extracted from the document
        """
        # Store the document in the corpus for TF-IDF
        self.document_corpus[doc_id] = text
        
        # If we have enough documents and using TF-IDF method, fit/refit the vectorizer
        if self.method == KeywordMethod.TFIDF and len(self.document_corpus) >= 2:
            self._fit_tfidf_vectorizer()
        
        # Extract keywords from the document
        keywords = self.extract_keywords(text)
        
        # Generate document embedding
        embedding = self.embedding_model.encode(text)
        
        # Generate a UUID based on doc_id for Qdrant
        point_id = self._get_uuid_from_string(doc_id)
        
        # Store document in vector DB with its keywords
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={"text": text[:1000], "keywords": keywords, "full_text_length": len(text), "original_id": doc_id}
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Error upserting document to vector DB: {e}")
        
        # Update global keyword tracking
        self.document_keywords[doc_id] = set(keywords)
        
        # Update global keywords count
        for keyword in keywords:
            if keyword in self.global_keywords:
                self.global_keywords[keyword] += 1
            else:
                self.global_keywords[keyword] = 1
        
        # Update keyword to document mapping
        for keyword in keywords:
            if keyword not in self.keyword_to_docs:
                self.keyword_to_docs[keyword] = set()
            self.keyword_to_docs[keyword].add(doc_id)
        
        # Update corpus statistics for KP-Miner
        self.corpus_size = len(self.document_keywords)
        
        # Update document frequency for each term
        # Create candidate phrases from text
        stop_pattern = r'\b(' + r'|'.join(self.stop_words) + r')\b|[' + string.punctuation + r']'
        candidate_chunks = re.split(stop_pattern, text.lower())
        
        # Clean and extract valid terms
        extracted_terms = set()
        for chunk in candidate_chunks:
            if chunk and not chunk.isspace():
                chunk = re.sub(r'\s+', ' ', chunk).strip()
                
                if chunk and len(chunk) >= 3 and len(chunk.split()) <= 5:
                    extracted_terms.add(chunk)
        
        # Update document frequency counts
        for term in extracted_terms:
            if term in self.term_document_freq:
                self.term_document_freq[term] += 1
            else:
                self.term_document_freq[term] = 1
        
        # Prune global keywords if needed
        self._prune_global_keywords()
        
        # logger.info(f"Processed document {doc_id}: extracted {len(keywords)} keywords")
        return keywords
    
    def _get_uuid_from_string(self, input_string: str) -> str:
        """Convert a string to a valid UUID for Qdrant"""
        # If input is already a valid UUID, return it
        try:
            uuid_obj = uuid.UUID(input_string)
            return str(uuid_obj)
        except ValueError:
            # Hash the string to get a consistent UUID
            hash_obj = hashlib.md5(input_string.encode())
            return str(uuid.UUID(hash_obj.hexdigest()))
    
    def _prune_global_keywords(self):
        """Limit the global keyword set to max_keywords"""
        if len(self.global_keywords) > self.max_keywords:
            # Keep only the top N most frequent keywords
            sorted_keywords = sorted(self.global_keywords.items(), key=lambda x: x[1], reverse=True)
            self.global_keywords = dict(sorted_keywords[:self.max_keywords])
            
            # Update keyword_to_docs to only include keywords in global_keywords
            self.keyword_to_docs = {k: v for k, v in self.keyword_to_docs.items() 
                                   if k in self.global_keywords}
            
            # Update document_keywords to only include keywords in global_keywords
            for doc_id in self.document_keywords:
                self.document_keywords[doc_id] = {k for k in self.document_keywords[doc_id] 
                                                if k in self.global_keywords}
    
    def get_top_keywords(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top N keywords across all documents.
        
        Args:
            n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        sorted_keywords = sorted(self.global_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:n]
    
    def find_similar_documents(self, text: str, limit: int = 10) -> List[Dict]:
        """
        Find documents similar to the provided text.
        
        Args:
            text: Query text
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        # Generate query embedding
        query_vector = self.embedding_model.encode(text).tolist()
        
        # Search for similar documents
        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.payload.get("original_id", hit.id),
                    "score": hit.score,
                    "keywords": hit.payload.get("keywords", []),
                    "text_preview": hit.payload.get("text", "")[:200] + "..."
                })
                
            return results
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            return []
    
    def update_document_keywords(self, doc_id: str, new_text: str = None) -> List[str]:
        """
        Update keywords for an existing document, optionally with new text.
        
        Args:
            doc_id: The document identifier
            new_text: New document text (if None, re-process existing text)
            
        Returns:
            Updated list of keywords
        """
        if new_text is None:
            # Retrieve existing document
            point_id = self._get_uuid_from_string(doc_id)
            
            try:
                results = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True,
                    with_payload=True
                )
                
                if not results:
                    logger.warning(f"Document {doc_id} not found in vector DB")
                    return []
                    
                doc_data = results[0]
                text = doc_data.payload.get("text", "")
                
                # Check if we have full text or just a preview
                if len(text) == doc_data.payload.get("full_text_length", 0):
                    new_text = text
                else:
                    logger.warning(f"Only have text preview for {doc_id}, cannot reprocess without full text")
                    return list(self.document_keywords.get(doc_id, []))
            except Exception as e:
                logger.error(f"Error retrieving document {doc_id} from vector DB: {e}")
                return list(self.document_keywords.get(doc_id, []))
        
        # Remove old keyword associations
        if doc_id in self.document_keywords:
            old_keywords = self.document_keywords[doc_id]
            for keyword in old_keywords:
                self.global_keywords[keyword] -= 1
                if self.global_keywords[keyword] <= 0:
                    if keyword in self.global_keywords:
                        del self.global_keywords[keyword]
                    
                if keyword in self.keyword_to_docs:
                    self.keyword_to_docs[keyword].remove(doc_id)
                    if not self.keyword_to_docs[keyword]:
                        del self.keyword_to_docs[keyword]
        
        # Process document with new text
        return self.process_document(doc_id, new_text)
    
    def get_documents_with_keyword(self, keyword: str, limit: int = 10) -> List[str]:
        """
        Get documents that contain a specific keyword.
        
        Args:
            keyword: The keyword to search for
            limit: Maximum number of documents to return
            
        Returns:
            List of document IDs
        """
        if keyword not in self.keyword_to_docs:
            return []
            
        return list(self.keyword_to_docs[keyword])[:limit]
    
    def batch_process_documents(self, documents: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: Dictionary mapping document IDs to their text content
            
        Returns:
            Dictionary mapping document IDs to their extracted keywords
        """
        # First, store all documents in the corpus
        for doc_id, text in documents.items():
            self.document_corpus[doc_id] = text
            
        # If using TF-IDF and we have multiple documents, fit the vectorizer
        if self.method == KeywordMethod.TFIDF and len(self.document_corpus) >= 2:
            self._fit_tfidf_vectorizer()
            
        # Pre-process corpus statistics for all documents if using KP-Miner
        if self.method == KeywordMethod.KPMINER and len(documents) > 1:
            # Reset corpus stats
            self.term_document_freq = {}
            self.corpus_size = len(documents)
            
            # Process document frequency for each term in each document
            for doc_id, text in documents.items():
                # Create candidate phrases from text
                stop_pattern = r'\b(' + r'|'.join(self.stop_words) + r')\b|[' + string.punctuation + r']'
                candidate_chunks = re.split(stop_pattern, text.lower())
                
                # Clean and extract valid terms
                extracted_terms = set()
                for chunk in candidate_chunks:
                    if chunk and not chunk.isspace():
                        chunk = re.sub(r'\s+', ' ', chunk).strip()
                        
                        if chunk and len(chunk) >= 3 and len(chunk.split()) <= 5:
                            extracted_terms.add(chunk)
                
                # Update document frequency counts
                for term in extracted_terms:
                    if term in self.term_document_freq:
                        self.term_document_freq[term] += 1
                    else:
                        self.term_document_freq[term] = 1
                        
            logger.info(f"Pre-processed corpus statistics for {self.corpus_size} documents with {len(self.term_document_freq)} unique terms")
        
        # Now process each document
        results = {}
        for doc_id, text in tqdm(documents.items(), desc="Processing documents"):
            keywords = self.process_document(doc_id, text)
            results[doc_id] = keywords
        return results
    
    def save_state(self, filepath: str):
        """Save the current state to a file"""
        state = {
            "global_keywords": self.global_keywords,
            "document_keywords": self.document_keywords,
            "keyword_to_docs": self.keyword_to_docs,
            "max_keywords": self.max_keywords,
            "min_keyword_freq": self.min_keyword_freq,
            "method": self.method,
            "document_corpus": self.document_corpus,
            "tfidf_fitted": self.tfidf_fitted,
            "term_document_freq": self.term_document_freq,
            "corpus_size": self.corpus_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load state from a file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.global_keywords = state["global_keywords"]
        self.document_keywords = state["document_keywords"]
        self.keyword_to_docs = state["keyword_to_docs"]
        self.max_keywords = state["max_keywords"]
        self.min_keyword_freq = state["min_keyword_freq"]
        if "method" in state:
            self.method = state["method"]
        
        # Load document corpus if available
        if "document_corpus" in state:
            self.document_corpus = state["document_corpus"]
            
        # Load TF-IDF fitted flag if available
        if "tfidf_fitted" in state:
            self.tfidf_fitted = state["tfidf_fitted"]
            
        # Load corpus statistics if available
        if "term_document_freq" in state:
            self.term_document_freq = state["term_document_freq"]
        if "corpus_size" in state:
            self.corpus_size = state["corpus_size"]
            
        # If we loaded documents and are using TF-IDF, refit the vectorizer
        if self.method == KeywordMethod.TFIDF and len(self.document_corpus) >= 2:
            self._fit_tfidf_vectorizer()
            
        logger.info(f"Loaded state from {filepath}")
    
    def set_extraction_method(self, method: KeywordMethod):
        """Change the keyword extraction method"""
        self.method = method
        logger.info(f"Keyword extraction method changed to: {method.value}")
        
        # If switching to TF-IDF and we have documents, fit the vectorizer
        if method == KeywordMethod.TFIDF and len(self.document_corpus) >= 2:
            self._fit_tfidf_vectorizer()
    
    def _fit_tfidf_vectorizer(self):
        """Fit the TF-IDF vectorizer on all documents in the corpus"""
        try:
            # Get all document texts
            corpus_texts = list(self.document_corpus.values())
            
            # Fit the vectorizer
            self.tfidf_vectorizer.fit(corpus_texts)
            self.tfidf_fitted = True
            logger.info(f"Fitted TF-IDF vectorizer on {len(corpus_texts)} documents")
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {e}")
            self.tfidf_fitted = False
    
    def reprocess_all_documents(self) -> Dict[str, List[str]]:
        """
        Reprocess all documents with the current extraction method.
        
        Returns:
            Dictionary mapping document IDs to their updated keywords
        """
        results = {}
        
        # Retrieve all documents from vector DB
        # Note: This could be inefficient for large collections - consider batching
        try:
            scroll_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000  # Adjust based on expected collection size
            )
            points = scroll_results[0]
            
            # Clear current keyword tracking
            self.global_keywords = {}
            self.document_keywords = {}
            self.keyword_to_docs = {}
            
            # Reprocess each document
            for point in tqdm(points, desc="Reprocessing documents"):
                # Get original doc_id if available
                doc_id = point.payload.get("original_id", str(point.id))
                text = point.payload.get("text", "")
                
                if len(text) == point.payload.get("full_text_length", 0):
                    # We have full text
                    keywords = self.update_document_keywords(doc_id, text)
                    results[doc_id] = keywords
                else:
                    logger.warning(f"Only have text preview for {doc_id}, cannot reprocess")
                    
            return results
            
        except Exception as e:
            logger.error(f"Error reprocessing documents: {e}")
            return {}
    
    def get_document_keywords(self, doc_id: str, top_n: int = None) -> List[str]:
        """
        Get keywords from a specific document.
        
        Args:
            doc_id: The document identifier
            top_n: Maximum number of keywords to return (None returns all)
            
        Returns:
            List of keywords for the document
        """
        # Check if document exists in our records
        if doc_id not in self.document_keywords:
            logger.warning(f"Document ID '{doc_id}' not found in processed documents")
            return []
            
        # Get all keywords for the document
        keywords = list(self.document_keywords[doc_id])
        
        # Optionally limit to top N
        if top_n is not None and top_n > 0 and top_n < len(keywords):
            # Try to get the document from the vector DB to get ordered keywords
            point_id = self._get_uuid_from_string(doc_id)
            try:
                results = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True
                )
                
                if results:
                    # Get ordered keywords from the payload
                    ordered_keywords = results[0].payload.get("keywords", [])
                    return ordered_keywords[:top_n]
            except Exception as e:
                logger.warning(f"Could not retrieve ordered keywords for {doc_id}: {e}")
            
            # If we can't get ordered keywords, just return the first top_n
            return keywords[:top_n]
            
        return keywords 