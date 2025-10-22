"""
生成模組：把經過排序的文件片段組成提示，並呼叫 LLM 產生最終回覆。
重點保留來源、法域等資訊，讓回答具備溯源能力。
"""

from __future__ import annotations  # 支援型別標註引用未定義的類別。

from typing import Iterable, List  # Iterable 提供多型輸入，List 用於回傳片段清單。

from .data_models import AugmentedPrompt, SearchResult  # 匯入資料模型。
from .llm import BaseLLM  # 匯入 LLM 介面，讓生成流程可替換不同模型。
from .tracing import VerboseTracer  # 匯入 tracer，用於詳細記錄。


class AnswerGenerator:
    """AnswerGenerator 類別：封裝提示建構與回答生成流程。"""

    def __init__(
        self,
        llm: BaseLLM,  # 傳入的 LLM 實例，需符合 BaseLLM 介面。
        max_snippets: int = 3,  # 控制最多引用幾段上下文，避免超出上下文限制。
        tracer: VerboseTracer | None = None,  # 可選的 tracer，用於 verbose 模式。
    ) -> None:
        self.llm = llm  # 保存 LLM 實例以便 generate 時使用。
        self.max_snippets = max_snippets  # 保存最大片段數設定。
        self.tracer = VerboseTracer.ensure(tracer).bind("generator.py")  # 建立綁定 generator 模組的 tracer。
        self.tracer.log(f"初始化 AnswerGenerator：max_snippets={max_snippets}")  # 紀錄初始化資訊。

    def build_prompt(self, question: str, context: Iterable[SearchResult]) -> AugmentedPrompt:
        """
        build_prompt：根據選定的上下文組合 AugmentedPrompt。
        確保只取前 max_snippets 段，並記錄其 doc_id。
        """
        ordered_context = list(context)[: self.max_snippets]  # 將 context 轉成列表並截取指定數量。
        context_ids = [result.document.doc_id for result in ordered_context]  # 收集選用文件的 ID，方便紀錄。
        self.tracer.log(f"建構提示：question='{question}', 選取文件={context_ids}")  # 紀錄提示建構的內容。
        return AugmentedPrompt(question=question, context=ordered_context)  # 回傳 AugmentedPrompt 供生成使用。

    def generate(self, question: str, context: Iterable[SearchResult]) -> str:
        """
        generate：將上下文整理成提示後呼叫 LLM 產生回答。
        每段上下文會標註排名、來源與法域，以利後續追溯。
        """
        augmented = self.build_prompt(question, context)  # 先生成 AugmentedPrompt。
        snippets: List[str] = []  # 建立空列表收集格式化後的上下文片段。
        for rank, result in enumerate(augmented.context, start=1):  # 逐段處理上下文，並加入排名。
            doc = result.document  # 取得原始 Document。
            snippet = (  # 建立帶有排名與來源資訊的段落。
                f"[Rank {rank}] {doc.title}（來源：{doc.source}；法域：{doc.jurisdiction}）\n"
                f"{doc.content}"
            )
            snippets.append(snippet)  # 將格式化段落加入清單。
        self.tracer.log(f"共整理 {len(snippets)} 段上下文，準備交由 LLM 生成人答。")  # 紀錄整理好的段落數量。
        return self.llm.generate(augmented.question, snippets)  # 呼叫 LLM 生成回答並回傳。
