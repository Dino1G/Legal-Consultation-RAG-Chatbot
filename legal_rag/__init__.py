"""
Legal RAG 封裝介面：統一匯出套件對外可使用的類別與函式。
當使用 `from legal_rag import ...` 時，會從這份清單中載入成員。
"""

from .data_loader import load_documents  # 載入語料的工具函式。
from .factory import build_pipeline  # 建構完整 RAGPipeline 的工廠函式。
from .generator import AnswerGenerator  # 生成器類別。
from .llm import HuggingFaceLLM  # Hugging Face 語言模型包裝。
from .pipeline import PipelineOutput, RAGPipeline  # 管線主類別與輸出結構。
from .preprocessing import SimpleTokenizer  # Tokenizer。
from .ranker import LLMReranker  # 再排序器。
from .retriever import BM25Retriever  # 檢索器。
from .tracing import VerboseTracer  # Verbose 追蹤器。

__all__ = [
    "AnswerGenerator",  # 允許外部直接匯入生成器。
    "BM25Retriever",  # 允許外部直接匯入檢索器。
    "build_pipeline",  # 提供快速建構 Pipeline 的介面。
    "HuggingFaceLLM",  # 匯出 Hugging Face 語言模型包裝。
    "LLMReranker",  # 匯出再排序器。
    "PipelineOutput",  # 匯出 Pipeline 的輸出資料結構。
    "RAGPipeline",  # 匯出管線主體。
    "SimpleTokenizer",  # 匯出 tokenizer。
    "load_documents",  # 匯出資料載入函式。
    "VerboseTracer",  # 匯出追蹤工具。
]
