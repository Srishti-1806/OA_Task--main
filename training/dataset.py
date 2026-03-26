
import sys
sys.path.insert(0, ".")

from preprocessing.build_dataset import build_segments_dataset


def build_dataset():
    
    return build_segments_dataset()


if __name__ == "__main__":
    ds = build_dataset()
    print(f"Dataset size: {len(ds)}")
    print(f"Columns: {ds.column_names}")