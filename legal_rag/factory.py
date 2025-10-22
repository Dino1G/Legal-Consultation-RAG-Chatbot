"""
工廠模組：集中處理 RAGPipeline 的元件組裝，方便在單一函式中管理所有依賴。
可在此調整停用詞、權重或替換成不同的 LLM 與檢索器。
"""

from __future__ import annotations  # 允許型別標註引用尚未定義的類別。

from pathlib import Path  # Path 用於處理檔案路徑。

import os

from .generator import AnswerGenerator  # 匯入生成器。
from .llm import HuggingFaceLLM  # 匯入 Hugging Face LLM 實作。
from .pipeline import RAGPipeline  # 匯入管線類別。
from .preprocessing import SimpleTokenizer  # 匯入 tokenizer。
from .ranker import LLMReranker  # 匯入再排序器。
from .retriever import BM25Retriever  # 匯入檢索器。
from .data_loader import load_documents  # 匯入語料載入函式。
from .tracing import VerboseTracer  # 匯入 tracer，供 verbose 記錄使用。


def build_pipeline(corpus_path: str | Path, tracer: VerboseTracer | None = None) -> RAGPipeline:
    """
    build_pipeline：依據指定語料路徑建立完整的 RAGPipeline。
    調用順序：載入語料 → 建立 tokenizer → 建立 retriever/LLM/reranker/generator → 組合成 pipeline。
    """
    tracer = VerboseTracer.ensure(tracer)  # 確保 tracer 存在，即使呼叫端未提供也建立靜默 tracer。
    factory_tracer = tracer.bind("factory.py")  # 建立綁定 factory 模組的子 tracer。
    factory_tracer.log(f"開始建構 RAGPipeline，使用語料：{corpus_path}")  # 紀錄建構流程開始。

    documents = load_documents(corpus_path, tracer=tracer)  # 讀取語料，沿用同一 tracer 以保持完整追蹤。
    factory_tracer.log(f"載入文件完成，共 {len(documents)} 筆。")  # 紀錄語料數量。

    # 停用詞集合可依實務需求調整，這裡提供中英文常見停用詞。
    tokenizer = SimpleTokenizer(
        stop_words={"的", "與", "及", "and", "or", "the"},
        tracer=tracer,
    )
    retriever = BM25Retriever(documents, tokenizer=tokenizer, tracer=tracer)  # 建立 BM25 檢索器。
    hf_token = os.getenv("HF_TOKEN")
    hf_model_id = os.getenv("HF_MODEL_ID", "google/gemma-3-270m-it")
    hf_llm = HuggingFaceLLM(
        tokenizer=tokenizer,
        model_id=hf_model_id,
        hf_token=hf_token,
        tracer=tracer,
    )
    reranker = LLMReranker(llm=hf_llm, alpha=0.4, tracer=tracer)  # 利用 Llama 模型進行語義再排序。
    generator = AnswerGenerator(llm=hf_llm, max_snippets=3, tracer=tracer)  # 生成器直接使用同一語言模型。

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        tracer=tracer,
    )  # 將所有元件組裝成完整的 RAGPipeline。
    factory_tracer.log("RAGPipeline 建構完成。")  # 紀錄建構完成狀態。
    return pipeline  # 回傳組裝好的 pipeline 供 CLI 或其他服務使用。
