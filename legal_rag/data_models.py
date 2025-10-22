from __future__ import annotations  # 確保可以在型別標註中引用同檔案內的類別。

"""
資料模型模組：定義 RAG 各階段共用的資料結構。所有流程會以這些 dataclass 交換資訊。
"""

from dataclasses import dataclass  # dataclass 讓我們用簡潔語法定義只包含資料的類別。
from typing import List, Optional  # 匯入 List 與 Optional 表示欄位型別。


@dataclass(frozen=True)  # frozen=True 使實例不可變，避免流程中被意外修改。
class Document:
    """
    Document 類別：封裝單一法律文件的所有欄位。
    doc_id：資料庫中的識別碼。
    title：文件標題，方便顯示。
    content：文件內容（可含多語）。
    source：資料來源，例如法律法規名稱。
    jurisdiction：適用法域，方便篩選。
    tags：主題標籤，用於後續擴充分類。
    """

    doc_id: str  # 文件識別碼，用於追蹤來源。
    title: str  # 文件標題，供 UI 或回答顯示。
    content: str  # 文件正文，檢索與生成都依賴此欄位。
    source: str  # 紀錄資料來源，例如 Civil Code of Taiwan。
    jurisdiction: str  # 司法管轄區，協助多地法規管理。
    tags: List[str]  # 主題標籤，可用於快速搜尋或過濾。


@dataclass(frozen=True)
class SearchResult:
    """
    SearchResult 類別：用於描述檢索與再排序後的單一結果。
    document：對應的 Document 物件。
    score：系統計算出的分數（可混合 BM25 與語義分數）。
    reason：可選的字串，記錄分數來源或解釋。
    """

    document: Document  # 原始 Document 參考，方便後續生成引用。
    score: float  # 最終分數，供排序與展示使用。
    reason: Optional[str] = None  # 追蹤分數形成原因，預設可為 None。


@dataclass(frozen=True)
class AugmentedPrompt:
    """
    AugmentedPrompt 類別：生成階段的輸入容器。
    question：使用者問題原文。
    context：排序後的 SearchResult 清單，用於組合提示。
    """

    question: str  # 使用者的法律問題，生成時需要引用。
    context: List[SearchResult]  # 提供給生成器的上下文列表。
