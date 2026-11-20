import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from pymongo import MongoClient


def main():
	dotenv_path = Path(__file__).resolve().parent.parent / ".env"
	print("Reading .env from:", dotenv_path)
	print("Values read from .env:", dict(dotenv_values(dotenv_path)))
	load_dotenv(dotenv_path)
	uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI") or os.environ.get("MONGO_DB_URI")
	print("MongoDB URI:", uri)
	collection_name = os.environ.get("MONGO_COLLECTION")
	if not uri or not collection_name:
		raise SystemExit("Set MONGO_URI, MONGODB_URI, or MONGO_DB_URI and MONGO_COLLECTION")

	with MongoClient(uri) as client:
		database = client.get_default_database()
		if database is None:
			raise SystemExit("The MongoDB URI must include a database name")
		document = database[collection_name].find_one()

	if document is None:
		print("No documents found")
	else:
		print(document)


if __name__ == "__main__":
	main()
