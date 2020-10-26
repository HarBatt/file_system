from elasticsearch import Elasticsearch

class Index:
    def __init__(self):
        self.__mappings = {"mappings": {"properties": {"meta_data": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, "embedding_vector": {"type": "dense_vector", "dims": 1280}}}}
    

    def create_index(self, name):
        es = Elasticsearch(HOST = 'https://localhost', PORT = 9200)
        es.indices.create(index=name, ignore=400, body= self.__mappings)


es = Index()
es.create_index("nis_review")
