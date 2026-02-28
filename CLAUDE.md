# 场景1：报价单自动化

## 功能
外贸业务员输入客户询价内容，系统通过向量检索价格库，30秒内自动生成报价单。

## 技术栈
- FastAPI + uvicorn（端口 8081）
- ChromaDB 向量数据库（本地持久化）
- ZhipuAI embedding-3 向量化 + GLM-4-flash 生成报价
- 前端：单页 HTML 内嵌在 main.py 的 `HTML` 字符串中

## 文件结构
```
main.py          主程序（API + HTML 全在这里）
requirements.txt
宏达五金.txt      价格库数据文件（启动时自动导入 ChromaDB）
永兴紧固件.txt
鑫源标准件.txt
chroma_db/       向量数据库（不进 git，首次运行自动生成）
.env             API Keys（不进 git）
```

## 本地启动
```bash
.venv/bin/python main.py
# 访问 http://localhost:8081/scene1
# Token: edwardluo
```

## API 路由
- `POST /scene1/api/quote` - 生成报价单（需 X-API-Key header）
- `GET/POST/DELETE /scene1/api/prices` - 价格库管理
- `GET /scene1` - 前端页面

## 修改 HTML 注意事项
HTML 内嵌在 `main.py` 底部的 `HTML = """..."""` 字符串中。
修改时注意：
- Python 字符串内的 `{` `}` 需要转义为 `{{` `}}`（除非是 JS 模板字符串里的 `${}`）
- 改完本地验证后再用 `deploy.sh scene1` 部署

## 待办
- [ ] 将 HTML 拆分为独立的 `index.html` 文件（提高可维护性）
