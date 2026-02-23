import os

def merge_files(output_path, chunk_prefix):
    output_dir = os.path.dirname(output_path)
    chunks = sorted([f for f in os.listdir(output_dir) if f.startswith(chunk_prefix) and ".part" in f])
    
    if not chunks:
        print(f"No chunks found for prefix: {chunk_prefix}")
        return

    print(f"Merging {len(chunks)} chunks into {output_path}...")
    with open(output_path, 'wb') as output_file:
        for chunk_name in chunks:
            chunk_path = os.path.join(output_dir, chunk_name)
            with open(chunk_path, 'rb') as f:
                output_file.write(f.read())
            print(f"Merged: {chunk_name}")
            
    print(f"Merge complete for {output_path}.")

if __name__ == "__main__":
    weights_dir = "/home/ubuntu/Janus/weights"
    
    # List of weight files to reconstruct
    weight_files = [
        "janus_init.pt",
        "janus_best.pt",
        "janus_final.pt"
    ]
    
    for weight_file in weight_files:
        output_path = os.path.join(weights_dir, weight_file)
        merge_files(output_path, weight_file + ".part")
