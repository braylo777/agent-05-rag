//! Retrieval-augmented answers over a small built-in knowledge base (BGE embeddings + Llama 3.3 on Workers AI).
use serde_json::{json, Value};
use std::cell::RefCell;

pub const MODEL: &str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
pub const EMBED: &str = "@cf/baai/bge-base-en-v1.5";
pub const SYSTEM: &str = "Answer ONLY from the numbered sources. Cite them like [1]. If the sources do not contain the answer, say you don't know.";
const DOCS_JSON: &str = include_str!("../data/docs.json");

pub fn docs() -> Vec<Value> {
    serde_json::from_str(DOCS_JSON).expect("docs.json")
}

#[allow(async_fn_in_trait)]
pub trait Ai {
    async fn run(&self, model: &str, input: Value) -> Result<Value, String>;
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut x, mut y) = (0.0, 0.0, 0.0);
    for (p, q) in a.iter().zip(b) {
        d += p * q;
        x += p * p;
        y += q * q;
    }
    let den = x.sqrt() * y.sqrt();
    d / if den == 0.0 { 1.0 } else { den }
}

pub fn top_k(qv: &[f64], vecs: &[Vec<f64>], k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = vecs.iter().enumerate().map(|(i, x)| (i, cosine(qv, x))).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); // stable
    v.truncate(k);
    v
}

pub fn text(r: &Value) -> String {
    match r.get("response") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => r.to_string(),
        Some(v) => v.to_string(),
    }
}

pub fn cap(s: Option<&str>, n: usize) -> bool {
    matches!(s, Some(s) if !s.trim().is_empty() && s.chars().count() <= n)
}

/// The document index is embedded once and cached per instance (per isolate in the Worker).
#[derive(Default)]
pub struct Rag {
    index: RefCell<Option<Vec<Vec<f64>>>>,
}

fn vecs(r: &Value) -> Vec<Vec<f64>> {
    r["data"].as_array().map(|a| a.iter().map(|v| v.as_array().map(|x| x.iter().map(|n| n.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default()).collect()).unwrap_or_default()
}

impl Rag {
    pub fn reset(&self) {
        *self.index.borrow_mut() = None;
    }
    pub async fn answer<A: Ai>(&self, ai: &A, question: &str, k: usize) -> Result<Value, String> {
        let docs = docs();
        if self.index.borrow().is_none() {
            let texts: Vec<String> = docs.iter().map(|d| format!("{}. {}", d["title"].as_str().unwrap_or(""), d["text"].as_str().unwrap_or(""))).collect();
            let v = vecs(&ai.run(EMBED, json!({"text": texts})).await?);
            *self.index.borrow_mut() = Some(v);
        }
        let qv = vecs(&ai.run(EMBED, json!({"text": [question]})).await?).into_iter().next().unwrap_or_default();
        let hits: Vec<(Value, f64)> = {
            let idx = self.index.borrow();
            top_k(&qv, idx.as_ref().unwrap(), k).into_iter().map(|(i, s)| (docs[i].clone(), (s * 1000.0).round() / 1000.0)).collect()
        };
        let context: Vec<String> = hits.iter().enumerate().map(|(n, (d, _))| format!("[{}] {}: {}", n + 1, d["title"].as_str().unwrap_or(""), d["text"].as_str().unwrap_or(""))).collect();
        let user = format!("Sources:\n{}\n\nQuestion: {}", context.join("\n"), question);
        let r = ai.run(MODEL, json!({"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":user}],"max_tokens":400})).await?;
        Ok(json!({"answer": text(&r), "sources": hits.iter().enumerate().map(|(n, (d, s))| json!({"ref": n + 1, "title": d["title"], "score": s})).collect::<Vec<_>>()}))
    }
}

/// Pure router: (status, json); status 0 = serve the page.
pub async fn handle<A: Ai>(rag: &Rag, ai: &A, method: &str, path: &str, body: &Value) -> (u16, Value) {
    if path == "/api/docs" {
        return (200, Value::Array(docs().iter().map(|d| json!({"id": d["id"], "title": d["title"]})).collect()));
    }
    if method == "POST" && path == "/api/ask" {
        let q = body.get("question").and_then(|v| v.as_str());
        if !cap(q, 500) {
            return (400, json!({"error": "question required (<=500 chars)"}));
        }
        return match rag.answer(ai, q.unwrap(), 3).await {
            Ok(v) => (200, v),
            Err(e) => (502, json!({"error": format!("Agent error: {e}")})),
        };
    }
    (0, Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::cell::{Cell, RefCell};

    struct Fake<F: Fn(&Value) -> Result<Value, String>>(F, Cell<usize>);
    impl<F: Fn(&Value) -> Result<Value, String>> Ai for Fake<F> {
        async fn run(&self, _m: &str, inp: Value) -> Result<Value, String> {
            if inp.get("text").map_or(false, |t| t.as_array().unwrap().len() > 1) {
                self.1.set(self.1.get() + 1);
            }
            (self.0)(&inp)
        }
    }
    fn fake<F: Fn(&Value) -> Result<Value, String>>(f: F) -> Fake<F> {
        Fake(f, Cell::new(0))
    }

    #[test]
    fn cosine_and_topk() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert!((cosine(&[2.0, 0.0], &[5.0, 0.0]) - 1.0).abs() < 1e-12);
        let ids = |v: Vec<(usize, f64)>| v.into_iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(ids(top_k(&[1.0, 0.0], &[vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]], 2)), [1, 2]);
        assert_eq!(ids(top_k(&[1.0, 0.0], &[vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]], 2)), [0, 1]);
    }
    #[test]
    fn answer_retrieves_and_cites() {
        let ai = fake(|inp| {
            if let Some(t) = inp.get("text") {
                let v = |s: &str| if s.contains("Model Context Protocol") || s.contains("MCP") { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
                return Ok(json!({"data": t.as_array().unwrap().iter().map(|x| v(x.as_str().unwrap())).collect::<Vec<_>>()}));
            }
            assert!(inp["messages"][1]["content"].as_str().unwrap().contains("[1] Model Context Protocol"));
            Ok(json!({"response": "MCP is a standard [1]."}))
        });
        let r = block_on(Rag::default().answer(&ai, "What is MCP?", 1)).unwrap();
        assert_eq!(r["sources"][0]["title"], "Model Context Protocol");
        assert!(r["answer"].as_str().unwrap().contains("[1]"));
        assert_eq!(docs().len(), 12);
    }
    #[test]
    fn docs_valid() {
        let d = docs();
        let ids: std::collections::HashSet<String> = d.iter().map(|x| x["id"].to_string()).collect();
        assert_eq!(ids.len(), d.len());
        assert!(d.iter().all(|x| !x["title"].as_str().unwrap().is_empty() && x["text"].as_str().unwrap().len() > 50));
    }
    #[test]
    fn index_embedded_once_and_sources_numbered() {
        let ai = fake(|inp| Ok(if let Some(t) = inp.get("text") { json!({"data": t.as_array().unwrap().iter().enumerate().map(|(i, _)| vec![1.0, i as f64 / 10.0]).collect::<Vec<_>>()}) } else { json!({"response": "x [1]"}) }));
        let rag = Rag::default();
        block_on(rag.answer(&ai, "a", 3)).unwrap();
        let r = block_on(rag.answer(&ai, "b", 3)).unwrap();
        assert_eq!(ai.1.get(), 1);
        let refs: Vec<i64> = r["sources"].as_array().unwrap().iter().map(|s| s["ref"].as_i64().unwrap()).collect();
        assert_eq!(refs, [1, 2, 3]);
        assert!(r["sources"][0]["score"].is_number());
    }
    #[test]
    fn http_docs_validation_and_502() {
        let ai = fake(|_| Err("quota".into()));
        let rag = Rag::default();
        let (s, d) = block_on(handle(&rag, &ai, "GET", "/api/docs", &json!({})));
        assert_eq!(s, 200);
        assert_eq!(d.as_array().unwrap().len(), 12);
        assert!(d[0].get("text").is_none());
        assert_eq!(block_on(handle(&rag, &ai, "POST", "/api/ask", &json!({}))).0, 400);
        assert_eq!(block_on(handle(&rag, &ai, "POST", "/api/ask", &json!({"question": "x".repeat(501)}))).0, 400);
        assert_eq!(block_on(handle(&rag, &ai, "POST", "/api/ask", &json!({"question": "hi"}))).0, 502);
        let _ = RefCell::new(0);
    }
}
