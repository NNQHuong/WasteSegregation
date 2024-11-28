import requests
import os

def download_file(file_id, destination):
  url = f'https://drive.google.com/uc?id={file_id}'
  response = requests.get(url)
  if response.status_code ==200:
    directory = os.path.dirname(destination)
    if directory:
      os.makedirs(directory, exist_ok=True)
    with open(destination, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
              f.write(chunk)
    return True
  else:
    return False

file_ids = [
    "1OsAa-MryW-4lNvzweL2xxw2qjn4yyUk4",
    "1HZlxtAUO1utwc2p9Rl5g5a54GJpNsdiS",
    "1BgyfGvFiqOpIWqHDH-FPnlbA5L-1TR7B"]
destination_paths = [
    "best.pt",
    "authorproject.jpg",
    "modelcomparison.jpg"]

for file_id, destination in zip(file_ids, destination_paths):
  if download_file(file_id, destination):
    print(f"Downloaded {destination}")
  else:
    print(f"Failed to download {destination}")


from ultralytics import solutions 
solutions.inference(model='best.pt')