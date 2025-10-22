"""
文字預處理模組：提供專為法律語料設計的簡易 tokenizer。
功能包含：正則擷取中英數字詞彙、中文拆成字與雙字組、停用詞過濾，以及 verbose 追蹤。
"""

from __future__ import annotations  # 允許在型別標註時引用未定義的型別。

import re  # re 模組用於建立正則表達式，擷取原始字串中的詞彙。
from typing import Iterable, List  # Iterable 用於停用詞參數，List 表示回傳型別。

from .tracing import VerboseTracer  # 匯入 VerboseTracer 以支援 verbose 模式記錄。


class SimpleTokenizer:
    """SimpleTokenizer 類別：負責將原始文字轉成 token 清單。"""

    _token_pattern = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")  # 正則模式：捕捉英數字與中日韓統一字元。

    def __init__(
        self,
        stop_words: Iterable[str] | None = None,  # stop_words：可選的停用詞集合來源。
        tracer: VerboseTracer | None = None,  # tracer：可選的追蹤器。
    ) -> None:
        self.stop_words = set(stop_words or [])  # 將停用詞轉成 set，加速查找；若無則使用空集合。
        self.tracer = VerboseTracer.ensure(tracer).bind("preprocessing.py")  # 建立綁定 preprocess 模組的 tracer。

    def tokenize(self, text: str) -> List[str]:
        """將輸入文字拆解為 token 清單，並記錄每一步驟。"""
        self.tracer.log(f"開始斷詞，輸入長度：{len(text)} 字元。")  # 紀錄輸入長度，方便 debug 長文本。
        raw_tokens = [match.group(0).lower() for match in self._token_pattern.finditer(text)]  # 使用 finditer 找出所有符合正則的詞並轉成小寫。
        self.tracer.log(f"初步擷取的 token：{raw_tokens}")  # 輸出初步擷取結果。
        tokens: List[str] = []  # 建立空列表，用於儲存最終 token。
        for token in raw_tokens:  # 逐一檢查每個初步 token。
            if token in self.stop_words:  # 如果 token 是停用詞，就忽略。
                # 遇到停用詞直接忽略，避免噪音影響 BM25 計分。
                continue
            if self._contains_cjk(token):  # 若 token 包含中文，需拆分成字與雙字組。
                # 中文詞彙拆分為字與連續雙字組，提高檢索成功率。
                tokens.extend(self._expand_cjk(token))  # 將拆出的字與雙字組加入輸出列表。
            else:
                tokens.append(token)  # 英文或數字詞彙直接保留。
        self.tracer.log(f"最終輸出 token：{tokens}")  # 紀錄最後輸出的 token 清單。
        return tokens  # 回傳 token 清單給上層模組使用。

    @staticmethod
    def _contains_cjk(token: str) -> bool:
        """檢查 token 是否包含中日韓統一表意文字，以決定是否需要拆分。"""
        # 判斷字串中是否包含中日韓統一表意文字區段的字元。
        return any("\u4e00" <= char <= "\u9fff" for char in token)

    @staticmethod
    def _expand_cjk(token: str) -> List[str]:
        """將中文 token 拆成單字與雙字組，提升中文檢索效果。"""
        chars = list(token)  # 將 token 拆成單一字元清單。
        bigrams = [token[i : i + 2] for i in range(len(token) - 1)]  # 建立連續雙字組。
        combined = chars + bigrams  # 合併單字與雙字組，保留原始順序。
        # Preserve order while removing duplicates.
        # 透過集合去除重複，同時保持原始順序，方便後續分析。
        seen = set()  # seen 集合用於判斷是否已加入。
        unique: List[str] = []  # unique 保持去重後的結果，維持原順序。
        for item in combined:  # 逐一檢查每個組合項目。
            if item not in seen:  # 若尚未加入此項目。
                seen.add(item)  # 標記為已出現。
                unique.append(item)  # 加入結果列表。
        return unique  # 回傳拆分後的字與雙字組。
