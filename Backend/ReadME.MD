# train
curl -X POST "http://127.0.0.1:8000/train" -F "images=@/path/to/image1.jpg" -F "images=@/path/to/image2.jpg" -F "labels=0" -F "labels=1"
# recognize
curl -X POST "http://127.0.0.1:8000/recognize" -F "image=@/path/to/image.jpg"
# save_model
curl -X POST "http://127.0.0.1:8000/save_model" -d '{"path": "/path/to/save/model.yml"}' -H "Content-Type: application/json"
# load_model
curl -X POST "http://127.0.0.1:8000/load_model" -d '{"path": "/path/to/save/model.yml"}' -H "Content-Type: application/json"
