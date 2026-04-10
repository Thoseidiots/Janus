use janus_core::{JanusState, IdentityContract};
use janus_llm::{MockLlmProvider, LlmProvider, Prompt};
use anyhow::Result;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- Janus Rust Core (Moltbook Integrated) ---");

    // 1. Initialize Identity Contract (Moltbook)
    let identity = IdentityContract {
        id: "JANUS-RUST-001".to_string(),
        creation_timestamp: Utc::now(),
        design_constraints: vec![
            "Identity lives outside model weights".to_string(),
            "Cognition happens in bounded cycles".to_string(),
            "Authority over substrate is externalized".to_string(),
        ],
        role_definition: "A persistent cognitive process whose identity is externalized, continuous, and bounded.".to_string(),
        non_negotiable_boundaries: vec![
            "Cannot change identity".to_string(),
            "Cannot redefine purpose".to_string(),
        ],
    };

    // 2. Initialize State
    let mut state = JanusState::new(identity);
    println!("[INIT] Identity loaded: {}", state.identity.id);

    // 3. Simulate an Observation
    state.add_event("observation", serde_json::json!({
        "input": "Hello Janus, what is your current state?",
        "source": "CLI"
    }));
    println!("[OBSERVE] Input received.");

    // 4. Simulate a Planning Cycle
    let task_id = state.add_task("Respond to user greeting and state inquiry.");
    println!("[PLAN] Task created: {}", task_id);

    // 5. Simulate LLM Reasoning
    let llm = MockLlmProvider;
    let prompt = Prompt {
        system: format!("Identity: {}", state.identity.role_definition),
        user: "Hello Janus, what is your current state?".to_string(),
    };
    
    let response = llm.generate(prompt).await?;
    println!("[PROPOSE] LLM Response: {}", response.content);

    // 6. Log the outcome
    state.add_event("response", serde_json::json!({
        "content": response.content,
        "task_id": task_id
    }));
    println!("[APPLY] Response logged to event stream.");

    println!("--- Janus Cycle Complete ---");
    Ok(())
}
