import requests

r = requests.post("http://127.0.0.1:8000/clean", params={"filename": "sample.csv"})
print("STATUS", r.status_code)
print("BODY", r.text)
