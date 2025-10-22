"""
檢索模組：採用經典 BM25 演算法，從法律語料中挑選與問題最相關的文件。
整個初始化流程會預先計算詞頻與文件統計，查詢時則根據 BM25 公式打分。
"""

from __future__ import annotations  # 允許型別註解引用尚未定義的類別。

import math  # math 提供對數計算，BM25 的 IDF 計算需要。
from collections import Counter, defaultdict  # Counter 計算詞頻，defaultdict 用於 DF 統計。
from typing import Dict, Iterable, List, Tuple  # 型別標註使用 Dict/Iterable/List/Tuple。

from .data_models import Document, SearchResult  # 匯入資料模型，方便建立輸出。
from .preprocessing import SimpleTokenizer  # 匯入 tokenizer，用於將文件與查詢轉成 token。
from .tracing import VerboseTracer  # 匯入 tracer，提供 verbose 執行紀錄。


class BM25Retriever:
    """BM25Retriever 類別：負責建立索引並提供基於 BM25 的檢索功能。"""

    def __init__(
        self,
        documents: Iterable[Document],  # 傳入的語料集合，可以是任何可迭代物件。
        tokenizer: SimpleTokenizer,  # 用於將文本轉成 token 的 tokenizer。
        k1: float = 1.5,  # BM25 的調整參數 k1，控制詞頻飽和度。
        b: float = 0.75,  # BM25 的調整參數 b，控制文件長度正規化。
        tracer: VerboseTracer | None = None,  # 可選的 tracer，用於 verbose 模式。
    ) -> None:
        self.documents = list(documents)  # 將語料轉為列表，方便多次遍歷。
        if not self.documents:  # 檢查是否至少有一筆文件，避免 BM25 無法運作。
            raise ValueError("BM25Retriever requires at least one document.")  # 無文件時拋出錯誤。
        self.tokenizer = tokenizer  # 保存 tokenizer 供後續使用。
        self.k1 = k1  # 保存 k1 參數。
        self.b = b  # 保存 b 參數。
        self.tracer = VerboseTracer.ensure(tracer).bind("retriever.py")  # 建立綁定 retriever 的 tracer。
        self.tracer.log(f"初始化 BM25Retriever：文件數量 {len(self.documents)}, k1={k1}, b={b}")  # 紀錄初始化資訊。

        self._doc_token_freqs: List[Counter[str]] = []  # 每份文件的詞頻統計。
        self._document_frequency: Dict[str, int] = defaultdict(int)  # 全語料的文件頻率統計。
        self._doc_lengths: List[int] = []  # 每份文件的 token 數，用於長度正規化。

        for doc in self.documents:  # 逐一處理每份文件，建立索引資料。
            self.tracer.log(f"為文件 {doc.doc_id} 建立詞頻向量。")  # 紀錄目前處理的文件。
            tokens = self.tokenizer.tokenize(doc.content)  # 將文件內容轉成 token。
            freq = Counter(tokens)  # 計算該文件的詞頻。
            self._doc_token_freqs.append(freq)  # 儲存詞頻 Counter。
            self._doc_lengths.append(len(tokens))  # 記錄文件長度（token 數）。
            for token in freq:  # 更新每個 token 的文件頻率。
                self._document_frequency[token] += 1  # 詞彙出現在該文件中，DF 加一。

        self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths)  # 計算平均文件長度，供 BM25 使用。
        self._idf_cache: Dict[str, float] = {}  # 建立 IDF 快取，避免重複計算。
        self.tracer.log(f"平均文件長度：{self._avg_doc_len:.2f}")  # 紀錄平均文件長度，便於檢查資料分佈。

    def _idf(self, token: str) -> float:
        """計算指定詞彙的逆文件頻率（IDF），並使用快取提升效能。"""
        if token in self._idf_cache:  # 若快取中已計算過，直接回傳。
            return self._idf_cache[token]
        df = self._document_frequency.get(token, 0)  # 取得該詞出現於多少文件。
        numerator = len(self.documents) - df + 0.5  # BM25 IDF 分子，含平滑項 0.5。
        denominator = df + 0.5  # BM25 IDF 分母，同樣加入 0.5 平滑。
        value = math.log((numerator / denominator) + 1)  # 使用 log-smoothing，避免 0 值或極大值。
        self._idf_cache[token] = value  # 將計算結果寫入快取。
        self.tracer.log(f"計算 IDF：token='{token}', df={df}, idf={value:.4f}")  # 紀錄 IDF 計算細節。
        return value  # 回傳 IDF。

    def _score(self, query_tokens: List[str], doc_index: int) -> float:
        """針對指定文件計算 BM25 分數，輸入為查詢 token 與文件索引。"""
        freq = self._doc_token_freqs[doc_index]  # 取得該文件的詞頻 Counter。
        score = 0.0  # 初始化分數累計。
        doc_len = self._doc_lengths[doc_index]  # 取得文件長度，用於長度正規化。
        for token in query_tokens:  # 逐一處理查詢中的 token。
            if token not in freq:  # 如果文件中沒有該詞，略過。
                continue
            idf = self._idf(token)  # 取得 token 的 IDF。
            numerator = freq[token] * (self.k1 + 1)  # BM25 分子：tf * (k1 + 1)。
            denominator = freq[token] + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)  # BM25 分母：tf 調整加上長度正規化。
            score += idf * (numerator / denominator)  # 累加該詞的分數貢獻。
            self.tracer.log(  # 紀錄每次累加的細節，方便追蹤分數來源。
                f"計分：doc_index={doc_index}, token='{token}', tf={freq[token]}, "
                f"idf={idf:.4f}, partial_score={score:.4f}"
            )
        return score  # 回傳文件的 BM25 總分。

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """針對使用者問題執行 BM25 檢索，回傳前 top_k 筆結果。"""
        self.tracer.log(f"接收查詢：{query}")  # 紀錄收到的查詢字串。
        query_tokens = self.tokenizer.tokenize(query)  # 將查詢轉成 token。
        self.tracer.log(f"查詢 token：{query_tokens}")  # 輸出查詢 token。
        scored: List[Tuple[int, float]] = []  # 建立空列表儲存 (文件索引, 分數)。
        for idx, _ in enumerate(self.documents):  # 逐一遍歷所有文件索引。
            score = self._score(query_tokens, idx)  # 計算該文件的 BM25 分數。
            if score > 0:  # 只保留正分文件，避免不相關結果。
                scored.append((idx, score))  # 加入分數列表。
                self.tracer.log(  # 紀錄文件得分，用於檢視排名結果。
                    f"文件 {self.documents[idx].doc_id} 取得分數 {score:.4f}"
                )
        scored.sort(key=lambda item: item[1], reverse=True)  # 依分數高低排序，最高分在前。

        results: List[SearchResult] = []  # 建立空列表準備轉成 SearchResult。
        for idx, score in scored[:top_k]:  # 取前 top_k 名的文件。
            doc = self.documents[idx]  # 取得文件物件。
            reason = f"BM25 score {score:.4f}"  # 組合理由字串，包含 BM25 分數。
            self.tracer.log(f"入選：doc_id={doc.doc_id}, score={score:.4f}")  # 紀錄入選文件。
            results.append(SearchResult(document=doc, score=score, reason=reason))  # 建立 SearchResult 並加入列表。
        self.tracer.log(f"完成檢索，共回傳 {len(results)} 筆結果。")  # 檢索結束後紀錄回傳筆數。
        return results  # 回傳檢索結果列表。
