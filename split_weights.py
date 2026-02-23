import os

def split_file(file_path, chunk_size_mb=50):
    chunk_size = chunk_size_mb * 1024 * 1024
    file_name = os.path.basename(file_path)
    output_dir = os.path.dirname(file_path)
    
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            chunk_name = f"{file_name}.part{chunk_num:03d}"
            chunk_path = os.path.join(output_dir, chunk_name)
            
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            print(f"Created chunk: {chunk_name}")
            chunk_num += 1

if __name__ == "__main__":
    weights_path = "/home/ubuntu/Janus/weights/janus_init.pt"
    if os.path.exists(weights_path):
        split_file(weights_path)
    else:
        print(f"Error: {weights_path} not found.")
