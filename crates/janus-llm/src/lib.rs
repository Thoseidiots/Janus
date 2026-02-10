use async_trait::async_trait;
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub usage: Option<Usage>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[async_trait]
pub trait LlmProvider {
    async fn generate(&self, prompt: Prompt) -> Result<LlmResponse>;
}

pub struct MockLlmProvider;

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn generate(&self, _prompt: Prompt) -> Result<LlmResponse> {
        Ok(LlmResponse {
            content: "This is a mock response from the Janus LLM adapter.".to_string(),
            usage: Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        })
    }
}
