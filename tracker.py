import os
import sqlite3
from typing import List
#
from tqdm import tqdm


class Tracker:
    def __init__(
        self,
        output_dir: str = ".",
        batch_size: int = 1,
    ):
        self.batch_size = batch_size
        self.conn = sqlite3.connect(os.path.join(output_dir, "progress.db"))
        self.__create_db()

    def add_samples(self, sample_filepaths: List[str]):
        with self.conn:
            for sample in tqdm(sample_filepaths, total=len(sample_filepaths), desc="Indexing samples"):
                self.conn.execute(
                    "INSERT OR IGNORE INTO samples (path) VALUES (?)",
                    (sample,)
                )

    def get_batch(self) -> List[str]:
        with self.conn:
            cursor = self.conn.execute(
                "SELECT path FROM samples WHERE status='pending' LIMIT ?",
                (self.batch_size,)
            )
            return [row[0] for row in cursor.fetchall()]
        
    def mark_done(self, path: str):
        with self.conn:
            self.conn.execute(
                "UPDATE samples SET status='done' WHERE path=?",
                (path,)
            )

    def pending_count(self) -> int:
        with self.conn:
            cursor = self.conn.execute("SELECT COUNT(*) FROM samples WHERE status='pending'")
            return cursor.fetchone()[0]
        
    def close(self):
        self.conn.close()

    def __create_db(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    status TEXT CHECK(status IN ('pending', 'done', 'error')) DEFAULT 'pending'
                )
            """)

    
