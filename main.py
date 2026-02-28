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
HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <link rel="icon" type="image/png" href="/favicon.png">
  <link rel="apple-touch-icon" href="/favicon-192.png">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>报价单自动化 / Quotation Automation</title>
<script src="https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js" defer></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg-deep: #070b14;
  --bg-panel: #0d1424;
  --bg-card: #111827;
  --bg-hover: #1a2540;
  --border: #1e3a5f;
  --border-glow: #2a5298;
  --steel: #7ab8f5;
  --steel-light: #a8d4ff;
  --steel-dim: #4a90d9;
  --amber: #f5a623;
  --amber-dim: #c47d0e;
  --text-primary: #f0f6ff;
  --text-secondary: #b8d0ee;
  --text-muted: #6a8aaa;
  --success: #2ecc71;
  --danger: #e05555;
  --grid-color: rgba(74,144,217,0.06);
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'IBM Plex Sans',sans-serif;
  background-color:var(--bg-deep);
  color:var(--text-primary);
  min-height:100vh;
  background-image:
    linear-gradient(var(--grid-color) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid-color) 1px,transparent 1px);
  background-size:40px 40px;
  background-position:-1px -1px;
}
/* HEADER */
header{
  position:sticky;top:0;z-index:100;
  background:rgba(7,11,20,0.93);
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  box-shadow:0 1px 0 0 rgba(74,144,217,0.15),0 4px 24px rgba(0,0,0,0.4);
  padding:0 32px;height:64px;
  display:flex;align-items:center;justify-content:space-between;
}
.header-left{display:flex;align-items:center;gap:14px}
.header-icon{
  width:36px;height:36px;
  border:1px solid var(--steel);border-radius:6px;
  display:flex;align-items:center;justify-content:center;
  background:rgba(74,144,217,0.1);
  box-shadow:0 0 12px rgba(74,144,217,0.2);
}
.header-icon svg{width:20px;height:20px}
.header-title-zh{
  font-family:'Rajdhani',sans-serif;font-size:21px;font-weight:700;
  color:var(--text-primary);letter-spacing:0.05em;line-height:1;
}
.header-title-en{
  font-family:'IBM Plex Mono',monospace;font-size:12px;
  color:var(--steel);letter-spacing:0.15em;text-transform:uppercase;line-height:1;margin-top:2px;
}
.btn-portal{
  font-family:'IBM Plex Mono',monospace;font-size:13px;
  letter-spacing:0.1em;text-transform:uppercase;
  color:var(--amber);border:1px solid var(--amber-dim);
  background:transparent;padding:8px 18px;border-radius:4px;
  cursor:pointer;transition:all 0.2s;text-decoration:none;
  display:flex;align-items:center;gap:8px;
}
.btn-portal:hover{background:rgba(245,166,35,0.08);border-color:var(--amber);box-shadow:0 0 16px rgba(245,166,35,0.2)}
/* MAIN */
main{max-width:920px;margin:0 auto;padding:36px 24px 80px}
/* TABS */
.tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:28px}
.tab-btn{
  font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:600;
  letter-spacing:0.05em;color:var(--text-muted);
  background:transparent;border:none;border-bottom:2px solid transparent;
  padding:12px 28px;cursor:pointer;transition:all 0.2s;
  position:relative;bottom:-1px;
}
.tab-btn:hover{color:var(--text-secondary)}
.tab-btn.active{color:var(--steel-light);border-bottom-color:var(--steel)}
.tab-label-zh{display:block;font-size:17px}
.tab-label-en{display:block;font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:0.12em;opacity:0.7;margin-top:1px}
.tab-panel{display:none}
.tab-panel.active{display:block}
/* CARD */
.card{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:8px;padding:28px 32px;margin-bottom:20px;
}
.card-title{
  font-family:'Rajdhani',sans-serif;font-size:15px;font-weight:600;
  letter-spacing:0.2em;text-transform:uppercase;color:#c8e4ff;
  margin-bottom:20px;display:flex;align-items:center;gap:10px;
}
.card-title::before{
  content:'';display:inline-block;width:3px;height:14px;
  background:var(--amber);border-radius:2px;box-shadow:0 0 8px var(--amber);
}
/* FORM */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.form-group{display:flex;flex-direction:column;gap:7px}
label{
  font-family:'IBM Plex Mono',monospace;font-size:13px;
  letter-spacing:0.1em;color:#c0d8f0;
}
label span{color:var(--text-muted);font-size:12px;margin-left:6px}
input,select,textarea{
  background:var(--bg-panel);border:1px solid var(--border);
  border-radius:4px;color:var(--text-primary);
  font-family:'IBM Plex Sans',sans-serif;font-size:16px;
  padding:10px 14px;outline:none;
  transition:border-color 0.2s,box-shadow 0.2s;
  width:100%;
}
input:focus,select:focus,textarea:focus{
  border-color:var(--steel);box-shadow:0 0 0 2px rgba(74,144,217,0.15);
}
input::placeholder,textarea::placeholder{color:var(--text-muted)}
select option{background:var(--bg-panel)}
textarea{resize:vertical;min-height:90px}
.btn-primary{
  display:inline-flex;align-items:center;gap:6px;
  padding:9px 20px;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-size:14px;
  letter-spacing:0.08em;text-transform:uppercase;
  cursor:pointer;border:none;transition:all 0.2s;margin-top:8px;
  background:linear-gradient(135deg,var(--steel-dim),#1e4a8a);
  color:#fff;box-shadow:0 0 12px rgba(74,144,217,0.25);
}
.btn-primary:hover{
  background:linear-gradient(135deg,var(--steel),var(--steel-dim));
  box-shadow:0 0 20px rgba(74,144,217,0.4);
}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}
.btn-secondary{
  display:inline-flex;align-items:center;gap:6px;
  padding:9px 20px;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-size:14px;
  letter-spacing:0.08em;text-transform:uppercase;
  cursor:pointer;transition:all 0.2s;
  background:transparent;border:1px solid var(--border-glow);
  color:var(--text-secondary);
}
.btn-secondary:hover{border-color:var(--steel);color:var(--steel)}
.btn-danger{
  display:inline-flex;align-items:center;
  padding:4px 10px;border-radius:6px;
  font-family:'IBM Plex Mono',monospace;font-size:13px;
  letter-spacing:0.08em;cursor:pointer;transition:all 0.2s;
  background:transparent;border:1px solid rgba(224,85,85,0.4);
  color:var(--danger);
}
.btn-danger:hover{background:rgba(224,85,85,0.1)}
/* MSG */
.msg{padding:10px 14px;border-radius:4px;margin-top:12px;font-size:15px;font-family:'IBM Plex Mono',monospace}
.msg.err{background:rgba(224,85,85,0.1);border:1px solid rgba(224,85,85,0.3);color:#f08080}
.msg.ok{background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.25);color:var(--success)}
/* QUOTE RESULT */
.result-card{
  background:var(--bg-panel);border:1px solid var(--border-glow);
  border-radius:8px;padding:28px 32px;
  box-shadow:0 0 30px rgba(74,144,217,0.08);
  animation:fadeIn 0.35s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.result-header{
  display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--border);
}
.result-title{font-family:'Rajdhani',sans-serif;font-size:21px;font-weight:700}
.result-subtitle{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--text-muted);margin-top:4px;letter-spacing:0.08em}
.result-badge{
  background:rgba(46,204,113,0.1);border:1px solid rgba(46,204,113,0.3);
  color:var(--success);font-family:'IBM Plex Mono',monospace;
  font-size:13px;letter-spacing:0.1em;padding:4px 12px;border-radius:20px;
}
/* QUOTE TABLE */
.quote-table{width:100%;border-collapse:collapse;font-size:15px;margin-bottom:16px}
.quote-table thead tr{border-bottom:1px solid var(--border-glow)}
.quote-table th{
  font-family:'IBM Plex Mono',monospace;font-size:12px;
  letter-spacing:0.15em;text-transform:uppercase;
  color:var(--steel);padding:8px 12px;text-align:left;font-weight:500;
}
.quote-table td{
  padding:10px 12px;color:var(--text-secondary);
  border-bottom:1px solid rgba(30,58,95,0.4);
}
.quote-table tr:hover td{background:var(--bg-hover)}
.quote-table td.name{color:var(--text-primary);font-weight:500}
.quote-table td.price{font-family:'IBM Plex Mono',monospace;color:var(--amber)}
.quote-table td.tag span{
  background:rgba(74,144,217,0.12);border:1px solid rgba(74,144,217,0.25);
  color:var(--steel-light);font-family:'IBM Plex Mono',monospace;
  font-size:12px;padding:2px 8px;border-radius:10px;
}
.result-total{
  display:flex;justify-content:space-between;align-items:center;
  padding:14px 20px;background:rgba(74,144,217,0.06);
  border:1px solid rgba(74,144,217,0.2);border-radius:6px;
}
.result-total-label{font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:600;color:var(--text-secondary);letter-spacing:0.05em}
.result-total-value{font-family:'Rajdhani',sans-serif;font-size:32px;font-weight:700;color:var(--amber);text-shadow:0 0 20px rgba(245,166,35,0.3)}
.result-note{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--text-muted);margin-top:12px;letter-spacing:0.05em}
/* PRICE LIST */
.table-toolbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.table-info{font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--text-muted);letter-spacing:0.08em}
.price-table{width:100%;border-collapse:collapse;font-size:15px}
.price-table thead tr{border-bottom:1px solid var(--border-glow)}
.price-table th{
  font-family:'IBM Plex Mono',monospace;font-size:12px;
  letter-spacing:0.15em;text-transform:uppercase;
  color:var(--steel);padding:8px 12px;text-align:left;font-weight:500;
}
.price-table td{
  padding:11px 12px;color:var(--text-secondary);
  border-bottom:1px solid rgba(30,58,95,0.4);
}
.price-table tr:hover td{background:var(--bg-hover)}
.price-table td.name{color:var(--text-primary);font-weight:500}
.price-table td.price{font-family:'IBM Plex Mono',monospace;color:var(--amber)}
.price-table td.unit{font-family:'IBM Plex Mono',monospace;font-size:14px;color:var(--text-muted)}
/* ADD FORM */
.add-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
/* FOOTER */
footer{
  text-align:center;padding:24px;border-top:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;font-size:13px;
  color:var(--text-muted);letter-spacing:0.1em;
}
footer span{color:var(--steel)}
/* LOGIN OVERLAY */
.login-overlay{
  display:none;position:fixed;inset:0;
  background:rgba(0,0,0,0.7);backdrop-filter:blur(6px);
  z-index:9999;align-items:center;justify-content:center;
}
.login-box{
  background:var(--bg-card);border:1px solid var(--border-glow);
  border-radius:10px;padding:36px 32px;width:340px;
  box-shadow:0 0 40px rgba(74,144,217,0.15);
}
.login-title-zh{font-family:'Rajdhani',sans-serif;font-size:25px;font-weight:700;color:var(--text-primary);letter-spacing:0.05em}
.login-title-en{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--steel);letter-spacing:0.15em;text-transform:uppercase;margin-top:2px;margin-bottom:24px}
.login-err{color:#f08080;font-family:'IBM Plex Mono',monospace;font-size:14px;margin-bottom:10px;display:none}
.btn-login{
  width:100%;font-family:'Rajdhani',sans-serif;font-size:17px;font-weight:700;
  letter-spacing:0.1em;text-transform:uppercase;
  color:var(--bg-deep);background:var(--steel);
  border:none;border-radius:4px;padding:12px;cursor:pointer;
  transition:all 0.2s;margin-top:8px;
}
.btn-login:hover{background:var(--steel-light);box-shadow:0 0 20px rgba(74,144,217,0.4)}
@media(max-width:600px){
  header{padding:0 16px}
  main{padding:20px 14px 60px}
  .form-grid,.add-grid{grid-template-columns:1fr}
  .card{padding:18px 16px}
}

/* ── 场景简介侧栏 ─────────────────────────────────────────────────────────── */
.page-layout{display:flex;min-height:calc(100vh - 64px)}
.sidebar{width:260px;min-width:260px;background:rgba(10,18,35,0.95);border-right:1px solid rgba(122,184,245,0.2);padding:28px 20px;position:sticky;top:64px;height:calc(100vh - 64px);overflow-y:auto;transition:transform 0.3s}
.sidebar-title{font-family:'Rajdhani',sans-serif;font-size:15px;font-weight:600;color:#7ab8f5;letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid rgba(122,184,245,0.2)}
.sidebar-desc{font-family:'IBM Plex Sans',sans-serif;font-size:15px;color:rgba(200,220,255,0.75);line-height:1.7;margin-bottom:24px}
.sidebar-steps-title{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#f5a623;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px}
.sidebar-steps{list-style:none;padding:0;margin:0 0 24px}
.sidebar-steps li{font-family:'IBM Plex Sans',sans-serif;font-size:14px;color:rgba(200,220,255,0.7);padding:7px 0 7px 20px;border-bottom:1px solid rgba(122,184,245,0.07);position:relative;line-height:1.5}
.sidebar-steps li::before{content:counter(step);counter-increment:step;position:absolute;left:0;top:7px;width:14px;height:14px;background:rgba(122,184,245,0.15);border:1px solid rgba(122,184,245,0.3);border-radius:2px;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#7ab8f5;display:flex;align-items:center;justify-content:center;line-height:14px;text-align:center}
.sidebar-steps{counter-reset:step}
.sidebar-audience{font-family:'IBM Plex Sans',sans-serif;font-size:14px;color:rgba(200,220,255,0.55);line-height:1.6;padding:10px 12px;background:rgba(122,184,245,0.05);border-left:2px solid rgba(245,166,35,0.4);border-radius:0 4px 4px 0}
.sidebar-audience-label{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#f5a623;letter-spacing:1px;display:block;margin-bottom:4px}
.main-content{flex:1;min-width:0}
.sidebar-toggle{display:none;position:fixed;left:0;top:50%;transform:translateY(-50%);z-index:100;background:rgba(10,18,35,0.95);border:1px solid rgba(122,184,245,0.3);border-left:none;border-radius:0 6px 6px 0;padding:10px 6px;cursor:pointer;color:#7ab8f5;font-size:18px;writing-mode:vertical-rl}
@media(max-width:768px){
  .sidebar{position:fixed;left:0;top:0;height:100vh;z-index:200;transform:translateX(-100%)}
  .sidebar.open{transform:translateX(0)}
  .sidebar-toggle{display:flex}
  .page-layout{display:block}
}


/* anim */
.demo-steps{padding:16px 0}
.demo-step{display:flex;align-items:center;gap:12px;padding:8px 0;opacity:0.3;transition:opacity 0.3s}
.demo-step.active{opacity:1}
.demo-step.done{opacity:0.6}
.step-icon{font-size:18px;width:24px;text-align:center;flex-shrink:0}
.step-text{font-family:'IBM Plex Mono',monospace;font-size:14px;color:var(--steel-light)}
.step-bar{flex:1;height:3px;background:rgba(74,144,217,0.15);border-radius:2px;overflow:hidden}
.step-bar-fill{height:100%;width:0;background:var(--steel);border-radius:2px;transition:width 0.8s ease}
.demo-step.active .step-bar-fill{width:60%}
.demo-step.done .step-bar-fill{width:100%;background:var(--success)}
.step-spin{width:16px;height:16px;border:2px solid rgba(122,184,245,0.3);border-top-color:var(--steel);border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0;display:none}
.demo-step.active .step-spin{display:block}
.demo-step.done .step-spin{display:none}
@keyframes spin{to{transform:rotate(360deg)}}
/* cta */
.btn-cta-secondary{width:100%;padding:11px 20px;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:0.08em;text-transform:uppercase;cursor:pointer;background:transparent;border:1px solid var(--border-glow);color:var(--text-secondary);transition:all 0.2s;margin-top:16px}
.btn-cta-secondary:hover{border-color:var(--steel);color:var(--steel)}
.contact-card{display:none;margin-top:12px;padding:16px 20px;background:rgba(74,144,217,0.06);border:1px solid rgba(74,144,217,0.2);border-radius:6px;font-family:'IBM Plex Mono',monospace;font-size:14px;color:var(--text-secondary);line-height:2}
.contact-card.open{display:block}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="header-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="#7ab8f5" stroke-width="1.5">
        <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/>
        <rect x="9" y="3" width="6" height="4" rx="1"/>
        <path d="M9 12h6M9 16h4"/>
      </svg>
    </div>
    <div>
      <div class="header-title-zh">报价单自动化</div>
      <div class="header-title-en">Quotation Automation</div>
    </div>
  </div>
  <a href="/" class="btn-portal">&#8592; 返回门户 / Portal</a>
</header>

<button class="sidebar-toggle" id="sidebarToggle" onclick="document.getElementById('sidebar').classList.toggle('open')" title="场景简介">简介</button>
<div class="page-layout">
<aside class="sidebar" id="sidebar">
  <div class="sidebar-title">场景简介 / Overview</div>
  <p class="sidebar-desc">外贸业务员每天面对大量询价邮件，手动查价格、算利润、写报价单耗时费力。本系统通过向量检索价格库，输入客户询价内容，30秒内自动生成专业报价单。</p>
  <div class="sidebar-steps-title">使用步骤</div>
  <ol class="sidebar-steps">
    <li>在输入框填写客户询价内容（产品名称、规格、数量）</li>
    <li>点击「生成报价单」</li>
    <li>系统自动检索价格库并计算</li>
    <li>查看生成的报价单，确认后复制发送</li>
  </ol>
  <div class="sidebar-audience">
    <span class="sidebar-audience-label">适用对象</span>
    有固定价格体系、需要快速响应客户询价的外贸/制造业企业。
  </div>
</aside>
<div class="main-content">
<main>
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('quote',this)">
      <span class="tab-label-zh">智能报价</span>
      <span class="tab-label-en">Smart Quote</span>
    </button>
    <button class="tab-btn" onclick="switchTab('db',this)">
      <span class="tab-label-zh">价格库管理</span>
      <span class="tab-label-en">Price Database</span>
    </button>
  </div>

  <!-- TAB1 -->
  <div id="tab-quote" class="tab-panel active">
    <div class="card">
      <div class="card-title">客户询价 · Customer Inquiry</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--amber);margin-bottom:16px">原来需要 30 分钟，现在 3 分钟搞定</div>
      <div class="form-group">
        <label>询价内容 <span>/ Inquiry Details</span></label>
        <textarea id="inquiry" placeholder="例如：需要 M8×30 六角螺栓 500个，M8螺母 500个&#10;e.g. 500pcs M8×30 hex bolts, 500pcs M8 nuts"></textarea>
      </div>
      <button class="btn-primary" onclick="genQuote()">生成报价 / Generate Quote</button>
      <div id="quoteMsg"></div>
      <div class="demo-steps" id="demoSteps" style="display:none">
        <div class="demo-step" id="step1"><span class="step-icon">🔍</span><span class="step-text">向量化询价内容...</span><div class="step-bar"><div class="step-bar-fill"></div></div><div class="step-spin"></div></div>
        <div class="demo-step" id="step2"><span class="step-icon">⚡</span><span class="step-text">检索价格库匹配...</span><div class="step-bar"><div class="step-bar-fill"></div></div><div class="step-spin"></div></div>
        <div class="demo-step" id="step3"><span class="step-icon">📄</span><span class="step-text">AI 生成报价单...</span><div class="step-bar"><div class="step-bar-fill"></div></div><div class="step-spin"></div></div>
      </div>
    </div>
    <div id="quoteResult"></div>
    <button class="btn-cta-secondary" onclick="toggleContact()">📅 预约顾问演示 / Book a Demo</button>
    <div class="contact-card" id="contactCard">
      📞 联系顾问<br>
      微信：<strong>hd-hardware-sales</strong><br>
      邮箱：<strong>sales@hd-hardware.com</strong><br>
      Telegram：<strong>@hd_hardware</strong>
    </div>
  </div>

  <!-- TAB2 -->
  <div id="tab-db" class="tab-panel">
    <div class="card">
      <div class="card-title">添加产品 · Add Product</div>
      <div class="add-grid">
        <div class="form-group">
          <label>供应商 <span>/ Supplier</span></label>
          <input id="ns" placeholder="宏达五金">
        </div>
        <div class="form-group">
          <label>品名 <span>/ Product Name</span></label>
          <input id="nn" placeholder="六角螺栓">
        </div>
        <div class="form-group">
          <label>规格 <span>/ Spec</span></label>
          <input id="nsp" placeholder="M8×30">
        </div>
        <div class="form-group">
          <label>单位 <span>/ Unit</span></label>
          <input id="nu" placeholder="个">
        </div>
      </div>
      <div class="form-group" style="max-width:200px">
        <label>单价（元）<span>/ Unit Price</span></label>
        <input id="np" type="number" step="0.01" placeholder="0.12">
      </div>
      <button class="btn-primary" onclick="addPrice()" style="margin-top:16px">添加 / Add</button>
      <div id="addMsg"></div>
      <div style="margin-top:12px;font-family:'IBM Plex Mono',monospace;font-size:13px;color:var(--text-muted)">💡 支持 Excel 批量导入，初始数据我们协助迁移</div>
    </div>
    <div class="card">
      <div class="table-toolbar">
        <div class="card-title" style="margin-bottom:0">价格库 · Price Database</div>
        <div class="table-info" id="priceCount">加载中...</div>
      </div>
      <table class="price-table">
        <thead>
          <tr>
            <th>品名 / Product</th>
            <th>规格 / Spec</th>
            <th>单价 / Price</th>
            <th>单位 / Unit</th>
            <th>供应商 / Supplier</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="priceBody"><tr><td colspan="6" style="color:var(--text-muted);padding:20px 12px">加载中... / Loading...</td></tr></tbody>
      </table>
    </div>
  </div>
</main>
</div><!-- main-content -->
</div><!-- page-layout -->

<footer>Powered by <span>AI</span> · 智能制造解决方案 / Intelligent Manufacturing Solutions</footer>

<!-- LOGIN OVERLAY -->
<div class="login-overlay" id="loginOverlay">
  <div class="login-box">
    <div class="login-title-zh">&#128203; 报价单自动化</div>
    <div class="login-title-en">Quotation Automation System</div>
    <div class="login-err" id="loginErr">Token 错误，请重试 / Invalid token</div>
    <div class="form-group">
      <label>Access Token</label>
      <input type="password" id="loginTk" placeholder="输入访问令牌" onkeydown="if(event.key==='Enter')doLogin()">
    </div>
    <button class="btn-login" onclick="doLogin()">进入 / Enter</button>
  </div>
</div>

<script>
const TOKEN = localStorage.getItem('kiro_token') || '';
const H = {'X-API-Key': TOKEN, 'Content-Type': 'application/json'};

// Auth check
(function(){
  if (!localStorage.getItem('kiro_token')) {
    const el = document.getElementById('loginOverlay');
    el.style.display = 'flex';
  }
})();

function doLogin() {
  const t = document.getElementById('loginTk').value.trim();
  if (!t) return;
  fetch('/scene1/api/prices', {headers: {'X-API-Key': t}}).then(r => {
    if (r.ok) { localStorage.setItem('kiro_token', t); location.reload(); }
    else { document.getElementById('loginErr').style.display = 'block'; }
  }).catch(() => { document.getElementById('loginErr').style.display = 'block'; });
}

function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'db') loadPrices();
}

function runDemoSteps(onDone) {
  const steps = document.getElementById('demoSteps');
  steps.style.display = 'block';
  ['step1','step2','step3'].forEach(id => document.getElementById(id).classList.remove('active','done'));
  function activate(idx) {
    const ids = ['step1','step2','step3'];
    if (idx > 0) document.getElementById(ids[idx-1]).classList.replace('active','done');
    if (idx < ids.length) {
      document.getElementById(ids[idx]).classList.add('active');
      if (idx < 2) setTimeout(() => activate(idx+1), 800);
      else onDone();
    }
  }
  activate(0);
}
function finishDemoSteps() {
  document.getElementById('step3').classList.replace('active','done');
  setTimeout(() => { document.getElementById('demoSteps').style.display = 'none'; }, 600);
}
async function genQuote() {
  const inquiryEl = document.getElementById('inquiry');
  if (!inquiryEl.value.trim()) inquiryEl.value = '需要 M8×30 六角螺栓 500个，M8螺母 500个';
  const inquiry = inquiryEl.value.trim();
  const msgEl = document.getElementById('quoteMsg');
  const resultEl = document.getElementById('quoteResult');
  msgEl.innerHTML = '';
  resultEl.innerHTML = '';
  runDemoSteps(async () => {
    try {
      const r = await fetch('/scene1/api/quote', {method:'POST', headers:H, body:JSON.stringify({inquiry})});
      const d = await r.json();
      window._lastQuoteData = d;
      finishDemoSteps();
      if (!r.ok) { msgEl.innerHTML = `<div class="msg err">${d.detail}</div>`; return; }
      const rows = d.items.map(it => `
        <tr>
          <td class="name">${it.name}</td><td>${it.spec}</td>
          <td class="price">¥${it.unit_price}/${it.unit}</td>
          <td>${it.quantity}</td><td class="price">¥${it.total}</td>
          <td class="tag"><span>${it.supplier}</span></td>
        </tr>`).join('');
      resultEl.innerHTML = `
        <div class="result-card">
          <div class="result-header">
            <div>
              <div class="result-title">报价单 / Quotation</div>
              <div class="result-subtitle">QUO-${Date.now().toString().slice(-8)} · ${new Date().toLocaleDateString('zh-CN')}</div>
            </div>
            <div class="result-badge">✓ 匹配成功 / Matched</div>
          </div>
          <table class="quote-table">
            <thead><tr>
              <th>品名/Product</th><th>规格/Spec</th><th>单价/Price</th>
              <th>数量/Qty</th><th>小计/Sub</th><th>供应商/Supplier</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <div class="result-total">
            <div class="result-total-label">总报价 / Total Quote</div>
            <div class="result-total-value">¥${d.total_amount}</div>
          </div>
          ${d.note ? `<div class="result-note">备注 / Note: ${d.note}</div>` : ''}
          <div style="margin-top:16px;text-align:right">
            <button class="btn-secondary" onclick="exportXlsx()">&#11015; 导出 Excel / Export</button>
          </div>
        </div>`;
    } catch(e) {
      finishDemoSteps();
      msgEl.innerHTML = `<div class="msg err">请求失败 / Request failed: ${e.message}</div>`;
    }
  });
}
function exportXlsx() {
  const d = window._lastQuoteData;
  if (!d) return;
  const rows = [['品名','规格','单价(元)','单位','数量','小计(元)','供应商']];
  d.items.forEach(it => rows.push([it.name, it.spec, it.unit_price, it.unit, it.quantity, it.total, it.supplier]));
  rows.push([], ['总报价', '', '', '', '', d.total_amount, '']);
  if (d.note) rows.push(['备注', d.note]);
  const ws = XLSX.utils.aoa_to_sheet(rows);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, '报价单');
  XLSX.writeFile(wb, '报价单_' + new Date().toLocaleDateString('zh-CN').replace(/\//g,'-') + '.xlsx');
}
function toggleContact() {
  document.getElementById('contactCard').classList.toggle('open');
}

async function loadPrices() {
  const body = document.getElementById('priceBody');
  const count = document.getElementById('priceCount');
  try {
    const r = await fetch('/scene1/api/prices', {headers: H});
    const items = await r.json();
    count.textContent = `共 ${items.length} 条 / ${items.length} records`;
    if (!items.length) {
      body.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);padding:20px 12px">暂无数据 / No data</td></tr>';
      return;
    }
    body.innerHTML = items.map(it => `
      <tr>
        <td class="name">${it.name}</td>
        <td>${it.spec}</td>
        <td class="price">¥${it.price}</td>
        <td class="unit">${it.unit}</td>
        <td>${it.supplier}</td>
        <td><button class="btn-danger" onclick="delPrice('${it.id}')">删除</button></td>
      </tr>`).join('');
  } catch(e) {
    body.innerHTML = `<tr><td colspan="6" class="msg err">加载失败: ${e.message}</td></tr>`;
  }
}

async function addPrice() {
  ['ns','nn','nsp','nu','np'].forEach(id => {
    const el = document.getElementById(id);
    if (!el.value.trim()) el.value = el.placeholder;
  });
  const item = {
    supplier: document.getElementById('ns').value.trim(),
    name: document.getElementById('nn').value.trim(),
    spec: document.getElementById('nsp').value.trim(),
    unit: document.getElementById('nu').value.trim(),
    price: parseFloat(document.getElementById('np').value)
  };
  const msgEl = document.getElementById('addMsg');
  const r = await fetch('/scene1/api/prices', {method:'POST', headers:H, body:JSON.stringify(item)});
  if (r.ok) {
    msgEl.innerHTML = '<div class="msg ok">添加成功 / Added</div>';
    ['ns','nn','nsp','nu','np'].forEach(id => document.getElementById(id).value = '');
    loadPrices();
  } else {
    msgEl.innerHTML = '<div class="msg err">添加失败 / Failed</div>';
  }
}

function toggleContact_placeholder(){}
async function delPrice(id) {
  if (!confirm('确认删除？/ Confirm delete?')) return;
  await fetch(`/scene1/api/prices/${id}`, {method:'DELETE', headers:H});
  loadPrices();
}
</script>
</body>
</html>
"""



@app.get("/scene1", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081)
