from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["bank"]
invrecom_collection = db["invrecom_print"]

def save_result_to_db(result):
    invrecom_collection.insert_one({"result": result})
