use crate::daemon;
use eyre::WrapErr;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Serialize;
use serde_json::Value;

const DEFAULT_DAEMON_URL: &str = "http://127.0.0.1:3030/mcp";
const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

#[derive(Debug, Clone, Serialize)]
pub struct CliSuccess {
    pub ok: bool,
    pub tool: String,
    pub result: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct CliError {
    pub ok: bool,
    pub tool: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

pub async fn call_tool(
    url_override: Option<&str>,
    tool_name: &str,
    args: Value,
) -> eyre::Result<CliSuccess> {
    let url = resolve_tool_url(url_override);
    let client = reqwest::Client::new();
    let session_id = initialize_session(&client, &url).await?;
    send_initialized_notification(&client, &url, &session_id).await?;
    let value = call_tool_once(&client, &url, &session_id, tool_name, args).await?;
    Ok(CliSuccess {
        ok: true,
        tool: tool_name.to_string(),
        result: value,
    })
}

pub fn print_success(result: &CliSuccess) -> eyre::Result<()> {
    println!("{}", serde_json::to_string_pretty(result)?);
    Ok(())
}

pub fn print_error(err: &CliError) -> eyre::Result<()> {
    eprintln!("{}", serde_json::to_string_pretty(err)?);
    Ok(())
}

pub fn cli_error(tool: &str, message: impl Into<String>) -> CliError {
    CliError {
        ok: false,
        tool: tool.to_string(),
        message: message.into(),
        code: None,
        data: None,
    }
}

pub fn map_error(tool: &str, err: &eyre::Report) -> CliError {
    if let Some(mcp) = err.downcast_ref::<McpCliError>() {
        return CliError {
            ok: false,
            tool: tool.to_string(),
            message: mcp.message.clone(),
            code: Some(mcp.code),
            data: mcp.data.clone(),
        };
    }

    cli_error(tool, err.to_string())
}

fn resolve_tool_url(url_override: Option<&str>) -> String {
    if let Some(url) = url_override {
        return url.to_string();
    }

    if let Some(state) = daemon::read_state() {
        return format!("http://127.0.0.1:{}/mcp", state.port);
    }

    DEFAULT_DAEMON_URL.to_string()
}

async fn initialize_session(client: &reqwest::Client, url: &str) -> eyre::Result<String> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "vox-cli",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    });

    let response = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "application/json, text/event-stream")
        .body(body.to_string())
        .send()
        .await
        .wrap_err("failed to initialize daemon session")?;

    let session_id = response
        .headers()
        .get("mcp-session-id")
        .and_then(|v: &HeaderValue| v.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| eyre::eyre!("daemon response missing Mcp-Session-Id header"))?;

    let payload = response
        .text()
        .await
        .wrap_err("failed to read initialize response")?;
    parse_initialize_response(&payload)?;
    Ok(session_id)
}

async fn call_tool_once(
    client: &reqwest::Client,
    url: &str,
    session_id: &str,
    tool_name: &str,
    args: Value,
) -> eyre::Result<Value> {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args
        }
    });

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/json, text/event-stream"),
    );
    headers.insert(
        "mcp-session-id",
        HeaderValue::from_str(session_id).wrap_err("invalid session id header")?,
    );

    let response = client
        .post(url)
        .headers(headers)
        .body(body.to_string())
        .send()
        .await
        .wrap_err("failed to call daemon tool")?;
    let payload = response
        .text()
        .await
        .wrap_err("failed to read daemon tool response")?;
    parse_tool_response(&payload)
}

async fn send_initialized_notification(
    client: &reqwest::Client,
    url: &str,
    session_id: &str,
) -> eyre::Result<()> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("application/json, text/event-stream"),
    );
    headers.insert(
        "mcp-session-id",
        HeaderValue::from_str(session_id).wrap_err("invalid session id header")?,
    );

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    });

    client
        .post(url)
        .headers(headers)
        .body(body.to_string())
        .send()
        .await
        .wrap_err("failed to send initialized notification")?;

    Ok(())
}

fn parse_initialize_response(payload: &str) -> eyre::Result<()> {
    let value = first_json_message(payload)?;
    if value.get("error").is_some() {
        return Err(parse_mcp_error(value));
    }
    Ok(())
}

fn parse_tool_response(payload: &str) -> eyre::Result<Value> {
    let value = first_json_message(payload)?;
    if let Some(error) = value.get("error") {
        return Err(parse_mcp_error(error.clone()));
    }

    let result = value
        .get("result")
        .cloned()
        .ok_or_else(|| eyre::eyre!("daemon tool response missing result field"))?;
    Ok(normalize_tool_result(result))
}

fn first_json_message(payload: &str) -> eyre::Result<Value> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return Err(eyre::eyre!("empty daemon response"));
    }

    if trimmed.starts_with('{') {
        return serde_json::from_str(trimmed).wrap_err("invalid JSON response from daemon");
    }

    for block in trimmed.split("\n\n") {
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data);
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let joined = data_lines.join("\n");
        if joined.trim().is_empty() {
            continue;
        }
        return serde_json::from_str(&joined).wrap_err("invalid SSE JSON response from daemon");
    }

    Err(eyre::eyre!("no JSON message found in daemon response"))
}

fn normalize_tool_result(result: Value) -> Value {
    if let Some(text) = extract_single_text_content(&result)
        && let Ok(parsed) = serde_json::from_str::<Value>(&text)
    {
        return parsed;
    }
    result
}

fn extract_single_text_content(result: &Value) -> Option<String> {
    let content = result.get("content")?.as_array()?;
    if content.len() != 1 {
        return None;
    }
    let first = &content[0];
    if first.get("type")?.as_str()? != "text" {
        return None;
    }
    first.get("text")?.as_str().map(str::to_string)
}

fn parse_mcp_error(value: Value) -> eyre::Report {
    let code = value.get("code").and_then(|v| v.as_i64()).unwrap_or(-32603);
    let message = value
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown MCP error")
        .to_string();
    let data = value.get("data").cloned();
    eyre::eyre!(McpCliError {
        code,
        message,
        data,
    })
}

#[derive(Debug, thiserror::Error)]
#[error("MCP error {code}: {message}")]
struct McpCliError {
    code: i64,
    message: String,
    data: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_message_from_sse_payload() {
        let payload =
            "id: 0\nretry: 3000\ndata:\n\ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true}}\n\n";
        let value = first_json_message(payload).unwrap();
        assert_eq!(value["result"]["ok"], true);
    }

    #[test]
    fn normalize_tool_result_parses_single_text_json() {
        let value = serde_json::json!({
            "content": [{"type": "text", "text": "{\"queued\":true}"}],
            "isError": false
        });
        let normalized = normalize_tool_result(value);
        assert_eq!(normalized["queued"], true);
    }

    #[test]
    fn normalize_tool_result_keeps_non_json_text() {
        let value = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}],
            "isError": false
        });
        let normalized = normalize_tool_result(value.clone());
        assert_eq!(normalized, value);
    }

    #[test]
    fn parse_mcp_error_extracts_code_message_and_data() {
        let err = parse_mcp_error(serde_json::json!({
            "code": -32602,
            "message": "bad params",
            "data": {"field": "message"}
        }));
        let mcp = err.downcast_ref::<McpCliError>().unwrap();
        assert_eq!(mcp.code, -32602);
        assert_eq!(mcp.message, "bad params");
        assert_eq!(mcp.data.as_ref().unwrap()["field"], "message");
    }

    #[test]
    fn extract_single_text_content_returns_text() {
        let result = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}]
        });
        assert_eq!(
            extract_single_text_content(&result).as_deref(),
            Some("hello")
        );
    }
}
