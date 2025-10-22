"""
VerboseTracer 模組：提供簡易的階層式追蹤工具，協助在 verbose 模式下輸出每一個程式動作。
此檔案會被所有其他模組引用，以確保每一步驟都能追蹤到來源模組與詳細資訊。
"""

from __future__ import annotations  # 確保型別提示可引用尚未定義的類別名稱。

from dataclasses import dataclass  # 引入 dataclass 方便定義不可變的資料結構。
from typing import Sequence  # Sequence 用於描述 path 屬性的型別。


@dataclass(frozen=True)  # frozen=True 讓實例不可變，確保 trace 路徑不會被意外修改。
class VerboseTracer:
    """VerboseTracer 類別：封裝 verbose 模式需要的開關與路徑堆疊。"""

    enabled: bool = False  # enabled 表示是否啟用 verbose 輸出，預設關閉。
    path: Sequence[str] = ()  # path 紀錄目前 tracer 的層級路徑，預設為空 tuple。

    @staticmethod
    def ensure(tracer: "VerboseTracer | None") -> "VerboseTracer":
        """
        ensure 靜態方法：允許呼叫端傳入 None，如果沒有提供 tracer 就回傳一個停用輸出的預設 tracer。
        這樣其他模組就不需要額外判斷是否為 None，簡化程式碼。
        """
        if tracer is None:  # 如果呼叫端沒有提供 tracer，會進到這裡建立新的停用 tracer。
            return VerboseTracer(enabled=False)  # 回傳一個 enabled=False 的 tracer。
        return tracer  # 若已有 tracer，直接回傳原本的實例。

    def bind(self, segment: str) -> "VerboseTracer":
        """
        bind 方法：建立一個新的 tracer，路徑在原本 path 基礎上追加新的 segment。
        用於標記目前所在的模組或函式，使輸出訊息能顯示來源。
        """
        return VerboseTracer(enabled=self.enabled, path=(*self.path, segment))  # 回傳新的 tracer，保留 enabled 狀態並追加路徑。

    def log(self, message: str) -> None:
        """
        log 方法：在 verbose 模式開啟時輸出訊息。
        會根據 path 計算出前綴字串 prefix，格式為 [模組 > 子流程] message。
        """
        if not self.enabled:  # 如果沒有啟用 verbose，就直接返回不做任何事。
            return
        prefix = " > ".join(self.path) if self.path else "trace"  # 當 path 為空時使用預設 `trace` 作為前綴。
        print(f"[{prefix}] {message}")  # 實際印出追蹤訊息，格式符合要求的層級顯示。

