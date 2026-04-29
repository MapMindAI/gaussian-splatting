import sqlite3
import numpy as np
import cv2
import argparse
import os
import glob
from sklearn.cluster import MiniBatchKMeans
import joblib
import faiss
from tqdm import tqdm

RETRIEVAL_BOW_NUM_CLUSTERS = 1000
RETRIEVAL_BOW_KMEANS_PATH = "retrieval_vocab_kmeans.pkl"
RETRIEVAL_BOW_IDS_PATH = "retrieval_ids.npy"
RETRIEVAL_BOW_VECTORS_PATH = "retrieval_bow_vectors.npy"


def compute_word(kmeans, desc):
    word_ids = kmeans.predict(desc)
    hist, _ = np.histogram(word_ids, bins=np.arange(RETRIEVAL_BOW_NUM_CLUSTERS + 1))
    hist = hist.astype("float32") / np.linalg.norm(hist)
    return hist


class BowRetireval:
    def __init__(self, model_path, top_k=5):
        self.top_k = top_k
        self.kmeans = joblib.load(os.path.join(model_path, RETRIEVAL_BOW_KMEANS_PATH))
        self.image_ids = np.load(os.path.join(model_path, RETRIEVAL_BOW_IDS_PATH), allow_pickle=True)

        image_bow_vectors = np.load(os.path.join(model_path, RETRIEVAL_BOW_VECTORS_PATH))
        self.index = faiss.IndexFlatL2(image_bow_vectors.shape[1])
        self.index.add(image_bow_vectors)

    def retrieve_bow(self, desc):
        word_ids = self.kmeans.predict(desc)
        hist, _ = np.histogram(word_ids, bins=np.arange(RETRIEVAL_BOW_NUM_CLUSTERS + 1))
        hist = hist.astype("float32") / np.linalg.norm(hist)
        D, I = self.index.search(hist.reshape(1, -1), self.top_k)
        return [self.image_ids[i] for i in I[0]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", help="database name", default="database_3d.db", type=str)
    parser.add_argument("--model_path", help="model path", default="", type=str)
    parser.add_argument("--feature_dim", help="feature dimension", default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    db_path = os.path.join(args.model_path, args.database)

    sqlite_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sqlite_cursor = sqlite_conn.cursor()

    def get_descroptors(image_id):
        sqlite_cursor.execute("SELECT image_id, rows, cols, data FROM descriptors WHERE image_id = ?", (str(image_id),))
        result = sqlite_cursor.fetchone()
        assert result is not None
        desc = np.frombuffer(result[3], dtype=np.uint8).reshape((result[1], result[2]))
        return desc

    # Execute a SELECT query to get all rows from the table
    sqlite_cursor.execute(f"SELECT * FROM images;")
    rows = sqlite_cursor.fetchall()
    # Fetch all the rows
    all_descriptors = []
    progress_bar = tqdm(range(0, len(rows)), desc="Process Fetch All Descriptors")

    for row in rows:
        all_descriptors.append(get_descroptors(row[0]))
        progress_bar.update(1)
    progress_bar.close()

    all_descriptors = np.vstack(all_descriptors)  # shape: (N_total_kp, 256)
    print(" - ", all_descriptors.shape)

    print("Make Index")
    kmeans = MiniBatchKMeans(n_clusters=RETRIEVAL_BOW_NUM_CLUSTERS, batch_size=10000, verbose=1, n_init=3)
    kmeans.fit(all_descriptors)
    joblib.dump(kmeans, os.path.join(args.model_path, RETRIEVAL_BOW_KMEANS_PATH))

    progress_bar = tqdm(range(0, len(rows)), desc="Process Global Feature")

    image_ids = []
    image_bow_vectors = []
    for row in rows:
        word = compute_word(kmeans, get_descroptors(row[0]))
        image_bow_vectors.append(word)
        image_ids.append(row[0])
        progress_bar.update(1)

    progress_bar.close()
    np.save(os.path.join(args.model_path, RETRIEVAL_BOW_IDS_PATH), image_ids)
    np.save(os.path.join(args.model_path, RETRIEVAL_BOW_VECTORS_PATH), image_bow_vectors)

    print("Done!")
    sqlite_cursor.close()
    sqlite_conn.close()
