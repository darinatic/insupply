import numpy as np
import json
import openai
import torch
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

class BaseSemanticSearch:
    def __init__(self, data_file):
        """Initialize the base search class."""
        self.data = self._load_data(data_file)
        self.embeddings = None  
        self.model = None       

    def _load_data(self, data_file):
        """Load material data from a JSON file."""
        with open(data_file, 'r') as f:
            return json.load(f)

    def _generate_embeddings(self):
        """Generate embeddings for all descriptions. To be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement _generate_embeddings")

    def _cosine_similarity_to_percentage(self, cosine_similarity):
        """Convert cosine similarity score to a percentage (0-100%)."""
        return ((1 - cosine_similarity) * 100)

    def search(self, queries, top_k=5):
        """Perform semantic search for a list of queries."""
        query_embeddings = self._generate_query_embeddings(queries)
        distances = cdist(query_embeddings, self.embeddings, metric='cosine')
        results = []

        for i, query in enumerate(queries):
            ranked_indices = np.argsort(distances[i])[:top_k]
            matches = [
                {
                    "material_number": self.data[idx]['material_number'],
                    "description": self.data[idx]['description'],
                    "score": round(self._cosine_similarity_to_percentage(distances[i, idx]), 2)
                }
                for idx in ranked_indices
            ]
            results.append({"query": query, "matches": matches})
        return json.dumps(results)

    def _generate_query_embeddings(self, queries):
        """Generate embeddings for queries. To be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement _generate_query_embeddings")

class SentenceTransformerSearch(BaseSemanticSearch):
    def __init__(self, data_file, model_name='all-MiniLM-L6-v2'):
        super().__init__(data_file)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings for all descriptions using SentenceTransformer."""
        descriptions = [item['description'] for item in self.data]
        return self.model.encode(descriptions)

    def _generate_query_embeddings(self, queries):
        """Generate embeddings for queries using SentenceTransformer."""
        return self.model.encode(queries)

class BERTSearch(BaseSemanticSearch):
    def __init__(self, data_file, model_name='bert-base-uncased'):
        super().__init__(data_file)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings for all descriptions using BERT."""
        descriptions = [item['description'] for item in self.data]
        embeddings = []
        for text in descriptions:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)

    def _generate_query_embeddings(self, queries):
        """Generate embeddings for queries using BERT."""
        embeddings = []
        for text in queries:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)

class RoBERTaSearch(BaseSemanticSearch):
    def __init__(self, data_file, model_name='roberta-base'):
        super().__init__(data_file)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        descriptions = [item['description'] for item in self.data]
        embeddings = []
        for text in descriptions:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)
    
    def _generate_query_embeddings(self, queries):
        embeddings = []
        for text in queries:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)


class AdvancedSentenceTransformerSearch(BaseSemanticSearch):
    def __init__(self, data_file, model_name='all-MiniLM-L6-v2'):
        super().__init__(data_file)
        self.model_name = model_name  # Store the model name for later use
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings for all descriptions using SentenceTransformer."""
        descriptions = [item['description'] for item in self.data]
        return self.model.encode(descriptions)

    def _generate_query_embeddings(self, queries):
        """Generate embeddings for queries using SentenceTransformer."""
        return self.model.encode(queries)


from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFSearch(BaseSemanticSearch):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.model_name = 'TFIDF'
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        descriptions = [item['description'] for item in self.data]
        return self.vectorizer.fit_transform(descriptions).toarray()

    def _generate_query_embeddings(self, queries):
        return self.vectorizer.transform(queries).toarray()

class OpenAISearch(BaseSemanticSearch):
    def __init__(self, data_file, api_key, model_name="text-embedding-ada-002"):
        super().__init__(data_file)
        openai.api_key = api_key
        self.model_name = model_name
        self.model = model_name
        self.embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        descriptions = [item['description'] for item in self.data]
        embeddings = []
        for desc in descriptions:
            response = openai.Embedding.create(input=desc, model=self.model_name)
            embeddings.append(response['data'][0]['embedding'])
        return np.array(embeddings)

    def _generate_query_embeddings(self, queries):
        return np.array([
            openai.Embedding.create(input=query, model=self.model_name)['data'][0]['embedding']
            for query in queries
        ])


def evaluate_model(search_engine, test_data, top_k=5):
    """
    Evaluate a semantic search model on test data.
    
    Parameters:
        search_engine (BaseSemanticSearch): The semantic search model object.
        test_data (pd.DataFrame): DataFrame containing the test data.
        top_k (int): Number of top results to retrieve for each query.

    Returns:
        List[Dict]: A list of results containing evaluation details for each query.
    """
    results = []

    # Iterate through the test data
    for _, row in test_data.iterrows():
        query = row["Combined Description"]  
        expected_material_number = row["material_number"]  
        expected_description = row["description"]  

        # Retrieve the top matches using the search engine
        search_results = json.loads(search_engine.search([query], top_k=top_k))
        top_matches = search_results[0]["matches"]

        # Extract material numbers and similarity scores from the top matches
        retrieved_material_numbers = [match["material_number"] for match in top_matches]
        similarity_scores = [match["score"] for match in top_matches]

        # Check if the expected material number is in the top retrieved results
        is_correct = expected_material_number in retrieved_material_numbers

        # Append detailed results for this query
        results.append({
            "query": query,
            "expected": expected_material_number,
            "expected_description": expected_description,
            "retrieved_top_5": top_matches,  
            "retrieved_material_numbers": retrieved_material_numbers,
            "similarity_scores": similarity_scores,
            "is_correct": is_correct
        })

    return results
