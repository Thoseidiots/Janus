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
            
    print("Merge complete.")

if __name__ == "__main__":
    weights_path = "/home/ubuntu/Janus/weights/janus_init.pt"
    merge_files(weights_path, "janus_init.pt.part")
