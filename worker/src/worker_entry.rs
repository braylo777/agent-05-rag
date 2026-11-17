//! Cloudflare Worker entry (wasm32 only). Same routes and JSON as the old src/index.js.
use crate::{page::PAGE, rag::{handle, Ai, Rag}};
use serde_json::{json, Value};
use wasm_bindgen::JsCast;
use worker::*;

thread_local! { static RAG: std::rc::Rc<Rag> = std::rc::Rc::new(Rag::default()); }

struct CfAi(worker::Ai);
impl Ai for CfAi {
    async fn run(&self, model: &str, input: Value) -> std::result::Result<Value, String> {
        self.0.run::<Value, Value>(model, input).await.map_err(|e| e.to_string())
    }
}

async fn rate_ok(env: &Env, ip: &str) -> bool {
    let rl = match js_sys::Reflect::get(env.as_ref(), &"RATE_LIMITER".into()) {
        Ok(v) if !v.is_undefined() && !v.is_null() => v,
        _ => return true,
    };
    let opts = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&opts, &"key".into(), &ip.into());
    let Ok(f) = js_sys::Reflect::get(&rl, &"limit".into()).and_then(|f| f.dyn_into::<js_sys::Function>().map_err(|e| e)) else { return true };
    let Ok(p) = f.call1(&rl, &opts) else { return true };
    match wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(p)).await {
        Ok(r) => js_sys::Reflect::get(&r, &"success".into()).ok().and_then(|v| v.as_bool()).unwrap_or(true),
        Err(_) => true,
    }
}

#[event(fetch)]
async fn fetch(mut req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let path = req.url()?.path().to_string();
    let method = req.method();
    if method == Method::Post && path == "/api/ask" {
        let ip = req.headers().get("cf-connecting-ip")?.unwrap_or_else(|| "anon".into());
        if !rate_ok(&env, &ip).await {
            return Ok(Response::from_json(&json!({"error":"Rate limit: try again in a minute."}))?.with_status(429));
        }
    }
    let body: Value = if method == Method::Post { req.json().await.unwrap_or(json!({})) } else { json!({}) };
    let ai = CfAi(env.ai("AI")?);
    let rag = RAG.with(|r| r.clone());
    let (status, v) = handle(&rag, &ai, method.as_ref(), &path, &body).await;
    if status == 0 {
        let h = Headers::new();
        h.set("content-type", "text/html;charset=utf-8")?;
        return Ok(Response::ok(PAGE)?.with_headers(h));
    }
    Ok(Response::from_json(&v)?.with_status(status))
}
