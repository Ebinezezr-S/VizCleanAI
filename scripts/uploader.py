import requests

url = "http://127.0.0.1:8000/upload"
fp = r"D:\Document\Data Tool\vizclean_ds_app\data\uploads\sample.csv"
with open(fp, "rb") as f:
    r = requests.post(url, files={"file": f})
print(r.status_code)
print(r.text)
