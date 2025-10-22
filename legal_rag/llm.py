"""
LLM 模組：宣告 LLM 需要實作的介面，並提供 Hugging Face 模型的包裝。
這裡採用 meta-llama/Llama-3.2-1B-Instruct 做為預設的小型語言模型。
"""

from __future__ import annotations  # 允許型別註解提前引用類別。

from abc import ABC, abstractmethod  # 定義抽象基底類別與抽象方法。
from typing import Dict, Iterable, List  # 型別註解用。

import torch  # PyTorch 是 transformers 的基礎。
from huggingface_hub import login  # 用於程式化登入 Hugging Face。
from transformers import (  # 匯入 transformers 主要元件。
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .preprocessing import SimpleTokenizer  # 匯入 tokenizer。
from .tracing import VerboseTracer  # 匯入 tracer 以支援 verbose 模式。


class BaseLLM(ABC):
    """BaseLLM 抽象類別：定義排序與生成階段會用到的方法。"""

    @abstractmethod
    def score_relevance(self, question: str, candidate: str) -> float:
        """計算候選文本對於問題的語義相關程度。"""

    @abstractmethod
    def generate(self, question: str, context_snippets: Iterable[str]) -> str:
        """依據提供的上下文片段生成最終回答。"""


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFaceLLM 類別：包裝 Hugging Face 上的指令語言模型。

    - 預設使用 `meta-llama/Llama-3.2-1B-Instruct`。
    - `score_relevance` 會將問題與候選段落分別編碼成嵌入向量，再透過餘弦相似度評分。
    - `generate` 使用 prompt template 將上下文整理後交由模型生成回覆。
    """

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        model_id: str = "google/gemma-3-270m-it",
        hf_token: str | None = None,
        device: str | None = None,
        max_length: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        tracer: VerboseTracer | None = None,
    ) -> None:
        self.tracer = VerboseTracer.ensure(tracer).bind("llm.py::HuggingFaceLLM")
        self.tracer.log(f"初始化 HuggingFaceLLM，model_id={model_id}")

        if hf_token:
            self.tracer.log("偵測到 HF token，執行 huggingface_hub.login()")
            login(token=hf_token, new_session=False)

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.model.to(self.device)

        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.simple_tokenizer = tokenizer  # 用於輕量化文字前處理。

        self._embedding_cache: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # 嵌入計算與相似度評分
    # ------------------------------------------------------------------ #
    def _encode(self, text: str) -> torch.Tensor:
        """建立文本的平均池化嵌入，並使用快取減少重複工作。"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]  # 取最後一層隱藏狀態。
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            normalized = torch.nn.functional.normalize(pooled, dim=-1)

        embedding = normalized.squeeze(0).to("cpu")
        self._embedding_cache[text] = embedding
        return embedding

    def score_relevance(self, question: str, candidate: str) -> float:
        """使用餘弦相似度評估語義相關度。"""
        if not question.strip() or not candidate.strip():
            return 0.0
        q_emb = self._encode(question)
        c_emb = self._encode(candidate)
        score = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0)).item()
        self.tracer.log(f"score_relevance：cosine={score:.4f}")
        return score

    # ------------------------------------------------------------------ #
    # 回覆生成
    # ------------------------------------------------------------------ #
    def _build_prompt(self, question: str, snippets: List[str]) -> str:
        """組裝系統 / 使用者 prompt。"""
        bullet_points = "\n".join(f"- {snippet}" for snippet in snippets)
        prompt = (
            "你是一位熟悉台灣法規的法律助理，"
            "請根據提供的資料回答使用者問題，並保持專業、條理分明。\n\n"
            f"使用者問題：{question.strip()}\n\n"
            "可參考的資料：\n"
            f"{bullet_points}\n\n"
            "請以繁體中文回答，包含：\n"
            "1. 具體建議（條列式）\n"
            "2. 相關法律依據（若資料中有提供）\n"
            "3. 應注意事項或後續動作\n"
            "最後提醒讀者此為一般資訊並建議諮詢律師。\n"
        )
        return prompt

    def generate(self, question: str, context_snippets: Iterable[str]) -> str:
        """將上下文整理後交由模型生成回答。"""
        snippets = [snippet.strip() for snippet in context_snippets if snippet.strip()]
        if not snippets:
            return (
                "目前知識庫中沒有足夠的相關文本。"
                "請嘗試重新描述問題，或提供合約條款、發生事實等更多細節，以利進一步檢索。"
            )

        prompt = self._build_prompt(question, snippets)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = generated[len(prompt) :].strip()
        self.tracer.log(f"generate：輸出長度={len(answer)} 字元。")
        return answer
