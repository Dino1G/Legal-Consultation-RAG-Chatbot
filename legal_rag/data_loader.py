"""
資料載入模組：負責從 JSONL 檔案讀取法律語料並轉換成 Document 物件列表。
每個步驟都透過 VerboseTracer 紀錄詳盡動作，便於除錯與審查。
"""

from __future__ import annotations  # 允許在型別標註中引用稍後定義的類別名稱。

import json  # 用於將 JSON 字串解析成 Python 字典。
from pathlib import Path  # Path 物件提供跨平台的路徑操作。
from typing import List  # List 用於指定回傳型別。

from .data_models import Document  # 匯入 Document 資料結構。
from .tracing import VerboseTracer  # 匯入 tracer 以支援 verbose 模式。


def load_documents(path: str | Path, tracer: VerboseTracer | None = None) -> List[Document]:
    """
    load_documents 函式：讀取指定路徑的 JSONL 法律文件，輸出 Document 清單。
    path：字串或 Path，表示語料檔案位置。
    tracer：可選的 VerboseTracer，提供詳細的執行紀錄。
    """
    module_tracer = VerboseTracer.ensure(tracer).bind("data_loader.py")  # 建立綁定 data_loader.py 的 tracer，以便記錄來源。
    file_path = Path(path)  # 將輸入路徑標準化為 Path 物件，確保後續操作一致。
    module_tracer.log(f"開始載入語料檔案：{file_path}")  # 紀錄載入動作與檔案位置。
    if not file_path.exists():  # 檢查檔案是否存在，避免後續讀檔失敗。
        raise FileNotFoundError(f"Corpus file not found: {file_path}")  # 若不存在，拋出清楚的錯誤訊息。

    documents: List[Document] = []  # 預先建立空列表，準備收集所有 Document 物件。
    with file_path.open("r", encoding="utf-8") as f:  # 以 UTF-8 編碼開啟檔案，確保中英文都能正確解讀。
        for line_number, line in enumerate(f, start=1):  # 使用 enumerate 追蹤行號，start=1 方便人類閱讀。
            if not line.strip():  # strip() 移除空白字元，若為空字串代表是空行。
                # 跳過空行，避免產生無效資料。
                continue
            payload = json.loads(line)  # 將 JSON 字串轉為字典，方便提取欄位。
            module_tracer.log(  # 詳細記錄目前解析的行號與文件 metadata。
                f"解析第 {line_number} 行：doc_id={payload.get('id')}, title={payload.get('title')}"
            )
            documents.append(  # 把字典轉成 Document 物件並加入列表。
                Document(
                    doc_id=payload["id"],  # 必填欄位：文件 ID。
                    title=payload["title"],  # 必填欄位：標題。
                    content=payload["content"],  # 必填欄位：內容。
                    source=payload.get("source", "unknown"),  # 選填欄位：來源，預設 unknown。
                    jurisdiction=payload.get("jurisdiction", "unknown"),  # 選填欄位：法域，預設 unknown。
                    tags=list(payload.get("tags", [])),  # 選填欄位：標籤，確保為 list。
                )
            )
    module_tracer.log(f"完成載入，共 {len(documents)} 筆文件。")  # 讀取結束後，紀錄載入筆數以便驗證。
    return documents  # 回傳 Document 列表，供後續檢索器建立索引用。
