"""
排序模組：結合原始檢索分數與 LLM 語義分數，挑選最適合的上下文。
此模組提供 alpha 權重調整與詳細的 verbose 記錄，易於觀察每個候選的評分過程。
"""

from __future__ import annotations  # 允許型別標註引用尚未定義的類別。

from typing import Iterable, List  # Iterable 表示候選可以是任何可迭代物件，List 為回傳型別。

from .data_models import SearchResult  # 匯入 SearchResult，重新建立結果時會用到。
from .llm import BaseLLM  # 匯入 LLM 抽象介面，透過它計算語義分數。
from .tracing import VerboseTracer  # 匯入 tracer 以支援 verbose 模式。


class LLMReranker:
    """LLMReranker 類別：從候選集合中挑選前 top_k 名，並記錄每個步驟的分數。"""

    def __init__(
        self,
        llm: BaseLLM,  # 依賴注入的 LLM 物件，提供 score_relevance。
        alpha: float = 0.3,  # alpha 控制原始檢索分數的權重。
        tracer: VerboseTracer | None = None,  # 可選的 tracer，用於 verbose 記錄。
    ) -> None:
        if not 0.0 <= alpha <= 1.0:  # 確保 alpha 在合法範圍內，避免加權計算出錯。
            raise ValueError("alpha must be between 0 and 1")  # 超出範圍時拋出明確錯誤。
        self.llm = llm  # 保存 LLM 實例供 rerank 使用。
        self.alpha = alpha  # 保存 alpha。
        self.tracer = VerboseTracer.ensure(tracer).bind("ranker.py")  # 建立綁定 ranker 模組的 tracer。
        self.tracer.log(f"初始化 LLMReranker：alpha={alpha}")  # 紀錄初始化訊息，便於調參追蹤。

    def rerank(self, question: str, candidates: Iterable[SearchResult], top_k: int = 3) -> List[SearchResult]:
        """
        rerank 方法：傳入問題與檢索候選，回傳語義調整後的前 top_k 名 SearchResult。
        會對每個候選計算語義分數並與原 BM25 分數加權合併，同時保留原因。
        """
        reranked: List[SearchResult] = []  # 建立空列表用來存放重新評分的結果。
        for candidate in candidates:  # 逐一處理每個候選。
            snippet = f"{candidate.document.title}. {candidate.document.content}"  # 將標題與內容組合成一段文字供 LLM 評分。
            semantic_score = self.llm.score_relevance(question, snippet)  # 使用 LLM 計算語義分數。
            combined = self.alpha * candidate.score + (1 - self.alpha) * semantic_score  # 依據 alpha 權重計算最終分數。
            self.tracer.log(  # 詳細紀錄每個候選的分數構成。
                f"候選文件 {candidate.document.doc_id}：retrieval_score={candidate.score:.4f}, "
                f"semantic_score={semantic_score:.4f}, combined={combined:.4f}"
            )
            reranked.append(  # 建立新的 SearchResult，包含更新後的分數與理由。
                SearchResult(
                    document=candidate.document,  # 保留原始文件。
                    score=combined,  # 使用加權後的分數。
                    # reason 字段保留檢索分數與語義分數，方便稽核與除錯。
                    reason=f"{candidate.reason}; semantic {semantic_score:.4f}",
                )
            )
        reranked.sort(key=lambda result: result.score, reverse=True)  # 依最終分數排序，最高分在前。
        top_ids = [result.document.doc_id for result in reranked[:top_k]]  # 取出前 top_k 文件的 ID 供紀錄使用。
        self.tracer.log(f"重新排序完成，選出文件：{top_ids}")  # 紀錄最後選出的文件名單。
        return reranked[:top_k]  # 回傳前 top_k 的 SearchResult。
