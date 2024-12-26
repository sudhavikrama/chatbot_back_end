#Working Fine
import os
import time
from datetime import datetime
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from bson import ObjectId
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

# Load the embedding model
class EmbeddingModel:
    def __init__(self, model_name="nli-roberta-large"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def get_embedding(self, data):
        """Generates vector embeddings for the given data."""
        embedding = self.model.encode(data)
        return embedding.tolist()


# Handle MongoDB operations for Vector Search
class VectorSearch:
    def __init__(self, db_url, db_name, collection_name, index_name):
        self.client = MongoClient(db_url)
        self.collection = self.client[db_name][collection_name]
        self.index_name = index_name


    def create_search_index(self):
        """Create the search index in MongoDB."""
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": 1024,
                        "path": "embedding",
                        "similarity": "euclidean"
                    }
                ]
            },
            name=self.index_name,
            type="vectorSearch"
        )
        # Wait for initial sync to complete
        print("Polling to check if the index is ready. This may take up to a minute.")
        predicate = lambda index: index.get("queryable") is True
        while True:
            indices = list(self.collection.list_search_indexes(self.index_name))
            if len(indices) and predicate(indices[0]):
                break
            time.sleep(5)
        print(self.index_name + " is ready for querying.")

    def get_query_results(self, query, limit=10):
        """Gets results from a vector search query."""
        embedding = EmbeddingModel()
        query_embedding = embedding.get_embedding(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "exact": True,
                    "limit": limit
                }
            }, {
                "$project": {
                    "_id": 0,
                    "text": 1
                }
            }
        ]

        results = self.collection.aggregate(pipeline)
        return [doc for doc in results]


# Class to handle chatbot conversation history
class ChatBot:
    def __init__(self, db_url, db_name, collection_name, hf_token):
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.hf_token = hf_token
        os.environ["HF_TOKEN"] = self.hf_token
        self.llm = InferenceClient(
            "mistralai/Mistral-7B-Instruct-v0.3",
            token=self.hf_token
        )

    def get_chat_response(self, user_query, context_docs):
        context_string = " ".join([doc["text"] for doc in context_docs])
        prompt = f"""
        Use the following pieces of context to answer the question at the end.
        { context_string}
        Question: {user_query}
        """

        output = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024
        )

        return output.choices[0].message.content

    def store_conversation(self, user_query, response):
        conversation = {
            "user_query": user_query,
            "response": response,
            "liked": False,
            "disliked": False,
            "reason": "null",
            "timestamp": datetime.now()
        }
        result = self.collection.insert_one(conversation)
        chat_id = str(result.inserted_id)
        return chat_id


# Flask app class
class ChatBotApp:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.host = host
        self.port = port

        # Initialize components
        self.embedding_model = EmbeddingModel()
        self.vector_search = VectorSearch(
            db_url="mongodb+srv://projectworkdemo2020:gBoBQogr4VpMa9IB@cluster0.o19y3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true",
            db_name="rag_db2",
            collection_name="test4",
            index_name="vector_index1"
        )
        self.chatbot = ChatBot(
            db_url="mongodb+srv://projectworkdemo2020:gBoBQogr4VpMa9IB@cluster0.o19y3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true",
            db_name="rag_db3",
            collection_name="chat_history",
            hf_token="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"
        )
        # Restrict CORS to the front-end's domain
        front_end_url = "https://proud-beach-0b812fa10.4.azurestaticapps.net"
        CORS(self.app,origins=front_end_url)
        self._setup_routes()

    def _setup_routes(self):
        self.app.add_url_rule('/query', 'search_data', self.search_data, methods=['POST'])
        self.app.add_url_rule('/update-chat', 'update_chat_history', self.update_chat_history, methods=['POST'])
        self.app.add_url_rule('/','index',self.index)
    def index():
        print('Request for index page received')
       return render_template('index.html')

    def search_data(self):
        data = request.json
        user_query = data.get("user_input")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        context_docs = self.vector_search.get_query_results(user_query)
        response = self.chatbot.get_chat_response(user_query, context_docs)
        chat_id = self.chatbot.store_conversation(user_query, response)

        return jsonify({"response": response, "chat_id": chat_id})

    def update_chat_history(self):
        data = request.json
        chat_id = data.get('chat_id')

        if chat_id:
            try:
                object_id = ObjectId(chat_id)
            except Exception as e:
                return jsonify({'message': 'Invalid Chat ID format'}), 400

            result = self.chatbot.collection.update_one(
                {'_id': object_id},
                {
                    '$set': {
                        'liked': data.get('liked', False),
                        'disliked': data.get('disliked', False),
                        'reason': data.get('reason', None),
                    }
                }
            )

            if result.modified_count > 0:
                return jsonify({'message': 'Feedback updated successfully'})
            else:
                return jsonify({'message': 'No changes made or chat not found'}), 404
        else:
            return jsonify({'message': 'Chat ID is required'}), 400

    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)


# Initialize and run the app
if __name__ == '__main__':
    app = ChatBotApp()
    app.run()
