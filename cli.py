"""
CLI 介面模組：提供命令列互動式聊天功能，示範整個 RAG 法律諮詢流程。
包含參數解析、管線組裝以及輸出 verbose 紀錄等功能。
"""

from __future__ import annotations  # 允許型別註解引用尚未定義的類別。

import argparse  # argparse 用於解析命令列參數。
import json  # json 用來格式化輸出引用來源，方便閱讀。
from pathlib import Path  # Path 提供跨平台路徑操作。
from typing import Any, Dict  # Any 與 Dict 用於型別標註輸出字典。

from legal_rag.factory import build_pipeline  # 匯入 build_pipeline 以快速組裝 RAGPipeline。
from legal_rag.pipeline import RAGPipeline  # 匯入 RAGPipeline 型別，用於函式參數與型別提示。
from legal_rag.tracing import VerboseTracer  # 匯入 VerboseTracer 支援 verbose 模式。


def interactive_chat(pipeline: RAGPipeline, tracer: VerboseTracer) -> None:
    """
    interactive_chat：提供命令列互動式問答。
    pipeline：已組裝好的 RAGPipeline。
    tracer：用於追蹤聊天過程的 VerboseTracer。
    """
    tracer = VerboseTracer.ensure(tracer).bind("cli.py::interactive_chat")  # 建立綁定互動式聊天的 tracer。
    tracer.log("啟動互動式聊天。")  # 記錄聊天啟動。
    print("🚀 Legal RAG chatbot. Type 'exit' to quit.")  # 對使用者顯示提示訊息。
    while True:  # 進入無窮迴圈，直到使用者離開。
        try:
            question = input("\n法律問題> ").strip()  # 取得使用者輸入並去除前後空白。
        except (EOFError, KeyboardInterrupt):  # 捕捉 Ctrl+D 或 Ctrl+C。
            print("\nBye!")  # 顯示離開訊息。
            tracer.log("偵測到使用者中斷，關閉聊天。")  # 紀錄中斷事件。
            break  # 結束迴圈。
        if question.lower() in {"exit", "quit"}:  # 若輸入 exit 或 quit。
            print("Bye!")  # 顯示離開訊息。
            tracer.log("使用者輸入 exit，結束流程。")  # 紀錄離開原因。
            break  # 結束迴圈。
        if not question:  # 若輸入為空字串。
            print("請輸入問題或輸入 exit 離開。")  # 提醒使用者重新輸入。
            tracer.log("收到空字串輸入，提示使用者重新輸入。")  # 紀錄空輸入事件。
            continue  # 跳到下一輪迴圈。
        tracer.log(f"收到使用者問題：{question}")  # 紀錄使用者問題。
        output = pipeline.run(question)  # 呼叫 RAGPipeline 執行整個流程。
        tracer.log("Pipeline 執行完成，準備輸出結果。")  # 紀錄流程完成。
        print("\n--- 回覆 ---")  # UI 分隔線。
        print(output.answer)  # 顯示回答內容。
        print("\n--- 檢索來源 ---")  # 顯示引用來源標題。
        for ctx in output.context:  # 逐一列出使用到的上下文。
            doc = ctx.document  # 取得 Document 物件。
            metadata: Dict[str, Any] = {  # 整理輸出的引用資訊。
                "doc_id": doc.doc_id,
                "title": doc.title,
                "score": round(ctx.score, 4),
                "reason": ctx.reason,
                "source": doc.source,
            }
            print(json.dumps(metadata, ensure_ascii=False))  # 以 JSON 格式列印引用資訊。
            tracer.log(f"輸出引用來源：{metadata}")  # 紀錄引用資訊，方便除錯。
    tracer.log("互動式聊天已結束。")  # 聊天迴圈結束時紀錄。


def main() -> None:
    """main 函式：解析參數並啟動互動式聊天。"""
    parser = argparse.ArgumentParser(description="Legal consultation RAG chatbot")  # 建立參數解析器。
    parser.add_argument(  # 新增 --corpus 參數。
        "--corpus",
        type=str,
        default="data/legal_corpus.jsonl",
        help="Path to JSONL corpus file.",
    )
    parser.add_argument(  # 新增 --verbose 旗標。
        "--verbose",
        action="store_true",
        help="啟用 verbose 模式，輸出完整執行追蹤資訊。",
    )
    args = parser.parse_args()  # 解析命令列參數。

    root_tracer = VerboseTracer(enabled=args.verbose)  # 根據 --verbose 建立根 tracer。
    cli_tracer = root_tracer.bind("cli.py")  # 建立綁定 CLI 模組的 tracer。
    cli_tracer.log("解析參數完成，準備建構 Pipeline。")  # 紀錄參數解析完成。
    pipeline = build_pipeline(Path(args.corpus), tracer=root_tracer)  # 依指定語料建構 RAGPipeline。
    cli_tracer.log("Pipeline 建構完成，進入互動模式。")  # 紀錄 pipeline 準備就緒。
    interactive_chat(pipeline, tracer=root_tracer)  # 啟動互動式聊天。


if __name__ == "__main__":  # 只有直接執行此檔案時才會進入此區塊。
    main()  # 呼叫 main 函式開始程式流程。
