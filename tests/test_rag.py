import re
from agent05.rag import Rag, cosine, top_k, DOCS, handle


def test_cosine():
    assert cosine([1, 0], [1, 0]) == 1 and cosine([1, 0], [0, 1]) == 0
    assert cosine([0, 0], [1, 1]) == 0 and abs(cosine([2, 0], [5, 0]) - 1) < 1e-12


def test_top_k_ranks_and_ties_stable():
    assert [h["i"] for h in top_k([1, 0], [[0, 1], [1, 0], [1, 1]], 2)] == [1, 2]
    assert [h["i"] for h in top_k([1, 0], [[1, 0], [1, 0], [0, 1]], 2)] == [0, 1]


def test_rag_answer_retrieves_and_cites():
    vec = lambda t: [1, 0] if re.search(r"Model Context Protocol|MCP", t) else [0, 1]
    def ai(model, inp):
        if "text" in inp:
            return {"data": [vec(t) for t in inp["text"]]}
        assert re.search(r"\[1\] Model Context Protocol", inp["messages"][1]["content"])
        return {"response": "MCP is a standard [1]."}
    r = Rag(ai).answer("What is MCP?", 1)
    assert r["sources"][0]["title"] == "Model Context Protocol" and "[1]" in r["answer"] and len(DOCS) == 12


def test_docs_unique_ids_nonempty_text():
    assert len({d["id"] for d in DOCS}) == len(DOCS) and all(d["title"] and len(d["text"]) > 50 for d in DOCS)


def test_index_embedded_once():
    batches = []
    def ai(_m, inp):
        if "text" in inp:
            if len(inp["text"]) > 1:
                batches.append(1)
            return {"data": [[1, 0] for _ in inp["text"]]}
        return {"response": "x [1]"}
    rag = Rag(ai)
    rag.answer("a"); rag.answer("b")
    assert len(batches) == 1


def test_sources_carry_ref_numbers_and_scores():
    ai = lambda _m, inp: {"data": [[1, i / 10] for i in range(len(inp["text"]))]} if "text" in inp else {"response": "y"}
    r = Rag(ai).answer("q", 3)
    assert [s["ref"] for s in r["sources"]] == [1, 2, 3] and all(isinstance(s["score"], float) for s in r["sources"])


def test_http_docs_and_validation():
    rag = Rag(None)
    s, d = handle(rag, "GET", "/api/docs")
    assert s == 200 and len(d) == 12 and "text" not in d[0]
    assert handle(rag, "POST", "/api/ask", {})[0] == 400
    assert handle(rag, "POST", "/api/ask", {"question": "x" * 501})[0] == 400


def test_model_failure_is_502():
    def boom(*a):
        raise RuntimeError("quota")
    assert handle(Rag(boom), "POST", "/api/ask", {"question": "hi"})[0] == 502
