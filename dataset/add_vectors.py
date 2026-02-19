import pandas as pd
import pickle
import ast
import os

def parse_vector_string(vector_str):
    """Parse vector string like '[0.4, -1.4, 0.0]' to list of floats"""
    try:
        return ast.literal_eval(vector_str)
    except (ValueError, SyntaxError):
        print(f"Error parsing vector: {vector_str}")
        return None

def create_vector_mapping(csv_path):
    """Create mapping from mix_path to vector information"""
    df = pd.read_csv(csv_path)
    vector_mapping = {}

    for idx, row in df.iterrows():
        mix_path = row['mix']
        source1_vector = parse_vector_string(row['source1_vector'])
        source2_vector = parse_vector_string(row['source2_vector'])
        listener_position = parse_vector_string(row['listener_position'])

        if source1_vector is None or source2_vector is None or listener_position is None:
            print(f"Skipping row {idx+1} due to vector parsing error")
            continue

        vector_mapping[mix_path] = {
            'source1_vector': source1_vector,
            'source2_vector': source2_vector,
            'listener_position': listener_position,
            'room': row['room']
        }

    print(f"Created mapping for {len(vector_mapping)} samples")
    return vector_mapping

def add_vectors_to_pkl(pkl_path, vector_mapping, output_path=None):
    """Add vector information to PKL dataset"""
    if output_path is None:
        output_path = pkl_path

    # Load dataset
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    updated_count = 0
    skipped_count = 0

    # Add vectors to each sample
    for sample in dataset:
        mix_path = sample['mix_path']

        if mix_path in vector_mapping:
            vectors = vector_mapping[mix_path]
            sample['source1_vector'] = vectors['source1_vector']
            sample['source2_vector'] = vectors['source2_vector']
            sample['listener_position'] = vectors['listener_position']
            sample['room'] = vectors['room']
            updated_count += 1
        else:
            print(f"Warning: No vector info found for {mix_path}")
            skipped_count += 1

    # Save updated dataset
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Updated {updated_count} samples, skipped {skipped_count} samples")
    return updated_count, skipped_count

def main():
    csv_path = "mix_sources.csv"
    data_dir = "data"

    # Create vector mapping from CSV
    print("Loading vector mapping from CSV...")
    vector_mapping = create_vector_mapping(csv_path)

    # Process each PKL file
    pkl_files = ['train.pkl', 'valid.pkl', 'test.pkl']

    for pkl_file in pkl_files:
        pkl_path = os.path.join(data_dir, pkl_file)
        if os.path.exists(pkl_path):
            print(f"\nProcessing {pkl_file}...")
            updated, skipped = add_vectors_to_pkl(pkl_path, vector_mapping)
            print(f"Finished processing {pkl_file}: {updated} updated, {skipped} skipped")
        else:
            print(f"Warning: {pkl_path} not found")

    print("\nAll PKL files processed successfully!")

if __name__ == "__main__":
    main()
