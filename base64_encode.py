# run_this_locally.py
import base64

# Replace with the actual path to your secret file
filename = "data-processor.json"

with open(filename, "rb") as f:
    encoded_bytes = base64.b64encode(f.read())
    encoded_string = encoded_bytes.decode('utf-8')

print("Copy the following string into your Streamlit secret:")
print(encoded_string)