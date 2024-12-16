# import torch
import numpy as np
from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist
import json

# Load data and model
class SemanticSearch:
    def __init__(self, data_file, model_type='sentence_transformer'):
       
        self.data = self._load_data(data_file)
        self.model_type = model_type

#        if model_type == 'bert':
#            self.model_name = 'bert-base-uncased'
#            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#            self.model = AutoModel.from_pretrained(self.model_name)
#        else:
        
        self.model_type='sentence_transformer'
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)           

        self.embeddings = self._generate_embeddings()

    def _load_data(self, data_file):
        """Load material data from a JSON file."""
        import json
        with open(data_file, 'r') as f:
            return json.load(f)

    def _generate_embeddings(self):
        """Generate embeddings for all descriptions."""
        descriptions = [item['description'] for item in self.data]

#        if self.model_type == 'sentence_transformer':
        return self.model.encode(descriptions)
#        elif self.model_type == 'bert':
#            return self._generate_bert_embeddings(descriptions)

#    def _generate_bert_embeddings(self, texts):
#        """Generate embeddings using BERT."""
#        embeddings = []
#        for text in texts:
#            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#            with torch.no_grad():
#                outputs = self.model(**inputs)
#            # Use the [CLS] token representation as the embedding
#            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
#            embeddings.append(cls_embedding)
#        return np.array(embeddings)
    
    def _cosine_similarity_to_percentage(self, cosine_similarity):
        """Convert cosine similarity score to a percentage (0-100%)."""
        return ((1-cosine_similarity) * 100)

    def search(self, queries, top_k=5):
        """Perform semantic search for a list of queries."""
#        if self.model_type == 'sentence_transformer':
        query_embeddings = self.model.encode(queries)
#        elif self.model_type == 'bert':
#            query_embeddings = self._generate_bert_embeddings(queries)

        distances = cdist(query_embeddings, self.embeddings, metric='cosine')
        results = []

        for i, query in enumerate(queries):
            ranked_indices = np.argsort(distances[i])[:top_k]
            matches = [
                {
                    "material_number": self.data[idx]['material_number'],
                    "description": self.data[idx]['description'],
                   # "score": 1 - distances[i, idx],  # Cosine similarity (1 - distance)
                    "score": round(self._cosine_similarity_to_percentage(distances[i, idx]),2)
                }
                for idx in ranked_indices
            ]
            results.append({"query": query, "matches": matches})

        json_results = json.dumps(results)
        return json_results