from neural_loop_combiner.config import settings
from pymongo import MongoClient


def initialize_database(col, 
                        db = settings.MONGODB_DB,
                        server = settings.MONGODB_SERVER, 
                        port = settings.MONGODB_PORT, 
                        username = settings.MONGO_USERNAME, 
                        password = settings.MONGO_PASSWORD):
    return MongoClient(server, port, username=username, password=password)[db][col]