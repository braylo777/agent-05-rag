// Parity helper: same scripted questions as the legacy JS harness; prints JSON.
use agent_05_rag::rag::{Ai, Rag};
use futures::executor::block_on;
use serde_json::{json, Value};
const WORDS: [&str; 10] = ["react", "function", "retrieval", "embedding", "rank", "mcp", "agent", "memory", "guard", "eval"];
struct Fake;
impl Ai for Fake {
    async fn run(&self, _m: &str, inp: Value) -> Result<Value, String> {
        let vec = |t: &str| { let l = t.to_lowercase(); let mut v: Vec<f64> = WORDS.iter().enumerate().map(|(i, w)| if l.contains(w) { 1.0 + i as f64 / 100.0 } else { 0.0 }).collect(); v.push(0.05); v };
        if let Some(t) = inp.get("text") {
            return Ok(json!({"data": t.as_array().unwrap().iter().map(|x| vec(x.as_str().unwrap())).collect::<Vec<_>>()}));
        }
        Ok(json!({"response": format!("A: {}", inp["messages"][1]["content"].as_str().unwrap().encode_utf16().count())}))
    }
}
fn main() {
    let rag = Rag::default();
    let out: Vec<Value> = ["What does ReAct do?", "How are embedding vectors ranked?", "Tell me about MCP and agents", "zzz unknown", "memory and guardrails evals"].iter().map(|q| block_on(rag.answer(&Fake, q, 3)).unwrap()).collect();
    println!("{}", serde_json::to_string(&out).unwrap());
}
