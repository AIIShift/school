from fastapi import FastAPI
from pydantic import BaseModel

# 创建 FastAPI 实例
app = FastAPI()

# 定义请求体的模型（使用 Pydantic）
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

# 创建一个 GET 接口
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

# 创建一个 POST 接口
@app.post("/items/")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price}
