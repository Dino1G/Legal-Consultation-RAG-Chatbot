"""
管線模組：負責串接檢索、語義再排序與生成三個階段，形成完整的 RAG 流程。
外部只需呼叫 run 方法即可取得回答與引用來源。
"""

from __future__ import annotations  # 允許型別標註引用尚未定義的類別。

from dataclasses import dataclass  # dataclass 簡化輸出資料結構的定義。
from typing import List  # List 用於描述 context 型別。

from .data_models import SearchResult  # 匯入 SearchResult，run 會回傳此類型。
from .generator import AnswerGenerator  # 匯入生成器模組。
from .ranker import LLMReranker  # 匯入再排序模組。
from .retriever import BM25Retriever  # 匯入檢索模組。
from .tracing import VerboseTracer  # 匯入 tracer，提供 verbose 記錄。


@dataclass
class PipelineOutput:
    """PipelineOutput：封裝回答字串與上下文列表的資料結構。"""

    answer: str  # answer 字段保存最終 LLM 回覆。
    context: List[SearchResult]  # context 保存用於生成的 SearchResult 清單。


class RAGPipeline:
    """RAGPipeline 類別：提供 run 介面執行單輪 RAG 流程。"""

    def __init__(
        self,
        retriever: BM25Retriever,  # 檢索器，負責第一階段的文件搜尋。
        reranker: LLMReranker,  # 再排序器，負責語義加權。
        generator: AnswerGenerator,  # 生成器，產生最終回答。
        retrieval_top_k: int = 5,  # 檢索階段取回的文件數量，可依需求調整。
        tracer: VerboseTracer | None = None,  # 可選的 tracer，支援 verbose 模式。
    ) -> None:
        self.retriever = retriever  # 保存檢索器實例。
        self.reranker = reranker  # 保存再排序器實例。
        self.generator = generator  # 保存生成器實例。
        self.retrieval_top_k = retrieval_top_k  # 保存檢索階段的 top_k 設定。
        self.tracer = VerboseTracer.ensure(tracer).bind("pipeline.py")  # 建立綁定 pipeline 的 tracer。
        self.tracer.log(  # 初始化時記錄主要設定，方便檢查參數。
            "初始化 RAGPipeline："
            f"retrieval_top_k={retrieval_top_k}"
        )

    def run(self, question: str, answer_top_k: int = 3) -> PipelineOutput:
        """
        run 方法：執行完整的 RAG 階段並回傳 PipelineOutput。
        answer_top_k 控制再排序後提供給生成器的上下文數量。
        """
        self.tracer.log(f"開始處理問題：'{question}'")  # 紀錄開始處理的問題文字。
        retrieved = self.retriever.retrieve(question, top_k=self.retrieval_top_k)  # 執行 BM25 檢索。
        self.tracer.log(f"檢索完成，獲得 {len(retrieved)} 筆候選。")  # 紀錄檢索結果筆數。
        reranked = self.reranker.rerank(question, retrieved, top_k=answer_top_k)  # 進行語義再排序。
        self.tracer.log(f"排序完成，選出前 {len(reranked)} 筆作為生成上下文。")  # 紀錄選出的上下文數量。
        answer = self.generator.generate(question, reranked)  # 生成最終回覆。
        self.tracer.log("生成回答完成，回傳 PipelineOutput。")  # 紀錄流程完成。
        return PipelineOutput(answer=answer, context=reranked)  # 封裝回答與上下文後回傳。
