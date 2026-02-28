from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from zhipuai import ZhipuAI

# ── Config ────────────────────────────────────────────────────────────────────
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
GLM_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-3"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
CHROMA_PATH = Path(__file__).parent / "chroma_db"
DATA_PATH = Path(__file__).parent
FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"

API_KEY = "edwardluo"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_token(key: str = Depends(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return key

app = FastAPI()
zhipu = ZhipuAI(api_key=ZHIPU_API_KEY)
chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma.get_or_create_collection("price_library")


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed(text: str) -> list:
    resp = zhipu.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# ── 解析价格库 txt ─────────────────────────────────────────────────────────────
def parse_price_file(filepath: Path) -> list[dict]:
    items = []
    supplier = ""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("供应商："):
                supplier = line[4:]
            m = re.match(r"品名：(.+?)，规格：(.+?)，单位：(.+?)，单价：(.+?)元", line)
            if m:
                items.append({
                    "supplier": supplier,
                    "name": m.group(1),
                    "spec": m.group(2),
                    "unit": m.group(3),
                    "price": float(m.group(4)),
                })
    return items

# ── 初始化：导入价格库到 ChromaDB ──────────────────────────────────────────────
def init_price_library():
    if collection.count() > 0:
        return
    txt_files = list(DATA_PATH.glob("*.txt"))
    for f in txt_files:
        items = parse_price_file(f)
        for i, item in enumerate(items):
            text = f"{item['name']} {item['spec']} 供应商:{item['supplier']}"
            doc_id = f"{f.stem}_{i}"
            collection.add(
                ids=[doc_id],
                embeddings=[embed(text)],
                documents=[text],
                metadatas=[item],
            )

init_price_library()


# ── Models ────────────────────────────────────────────────────────────────────
class QuoteRequest(BaseModel):
    inquiry: str

class PriceItem(BaseModel):
    supplier: str
    name: str
    spec: str
    unit: str
    price: float

# ── Tab1：报价生成 ─────────────────────────────────────────────────────────────
@app.post("/scene1/api/quote")
async def generate_quote(req: QuoteRequest, _=Depends(verify_token)):
    query_vec = embed(req.inquiry)
    results = collection.query(query_embeddings=[query_vec], n_results=5)
    if not results["metadatas"][0]:
        raise HTTPException(status_code=404, detail="未找到匹配产品")

    candidates = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        candidates.append(
            f"品名:{meta['name']} 规格:{meta['spec']} 单价:{meta['price']}元/{meta['unit']} 供应商:{meta['supplier']}"
        )
    candidates_text = "\n".join(candidates)

    prompt = f"""你是一个外贸五金厂的报价助手。根据客户询价和候选产品，生成一份报价单。

客户询价：{req.inquiry}

候选产品：
{candidates_text}

请从候选产品中选择最匹配的产品，生成报价单。返回 JSON 格式：
{{
  "items": [
    {{"name": "产品名", "spec": "规格", "unit": "单位", "unit_price": 单价, "quantity": 数量, "total": 小计, "supplier": "供应商"}}
  ],
  "total_amount": 总金额,
  "note": "备注"
}}
JSON only, no markdown."""

    resp = zhipu.chat.completions.create(
        model=GLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail=f"LLM 返回格式错误: {raw}")
    return data


# ── Tab2：价格库管理 ───────────────────────────────────────────────────────────
@app.get("/scene1/api/prices")
async def list_prices(_=Depends(verify_token)):
    results = collection.get(include=["metadatas"])
    items = []
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        items.append({"id": doc_id, **meta})
    return items

@app.post("/scene1/api/prices")
async def add_price(item: PriceItem, _=Depends(verify_token)):
    import uuid
    doc_id = str(uuid.uuid4())
    text = f"{item.name} {item.spec} 供应商:{item.supplier}"
    collection.add(
        ids=[doc_id],
        embeddings=[embed(text)],
        documents=[text],
        metadatas=[item.model_dump()],
    )
    return {"id": doc_id, "status": "ok"}

@app.delete("/scene1/api/prices/{doc_id}")
async def delete_price(doc_id: str, _=Depends(verify_token)):
    collection.delete(ids=[doc_id])
    return {"status": "ok"}




# ── 前端 HTML ─────────────────────────────────────────────────────────────────
@app.get("/scene1", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
