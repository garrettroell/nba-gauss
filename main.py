from fastapi import FastAPI

app = FastAPI()

print('app loaded')

@app.get("/")
def read_root():
    print('inside of read root')
    return {"Hello": "World"}
