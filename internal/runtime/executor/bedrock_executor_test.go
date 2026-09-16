package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	_ "github.com/router-for-me/CLIProxyAPI/v7/internal/translator"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestBedrockExecutor_Identifier(t *testing.T) {
	exec := NewBedrockExecutor(&config.Config{})
	if exec.Identifier() != "bedrock" {
		t.Fatalf("expected bedrock, got %q", exec.Identifier())
	}
}

func TestBedrockUpstreamFormatSelection(t *testing.T) {
	cases := map[string]sdktranslator.Format{
		"global.anthropic.claude-sonnet-5":             sdktranslator.FormatClaude,
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0": sdktranslator.FormatClaude,
		"anthropic.claude-sonnet-5":                    sdktranslator.FormatClaude,
		"claude-sonnet-5":                              sdktranslator.FormatClaude,
		"openai.gpt-oss-120b-1:0":                      sdktranslator.FormatOpenAI,
		"openai.gpt-5.5":                               sdktranslator.FormatOpenAI,
		"meta.llama3-3-70b-instruct-v1:0":              sdktranslator.FormatOpenAI,
	}
	exec := NewBedrockExecutor(&config.Config{})
	for model, want := range cases {
		if got := exec.RequestToFormat(cliproxyexecutor.Request{Model: model + "(high)"}, cliproxyexecutor.Options{}); got != want {
			t.Errorf("RequestToFormat(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestBedrockTargetFromAuth(t *testing.T) {
	t.Run("runtime defaults", func(t *testing.T) {
		target := bedrockTargetFromAuth(&cliproxyauth.Auth{Attributes: map[string]string{
			"aws_profile": "dev",
			"aws_region":  "us-west-2",
		}})
		if target.baseURL != "https://bedrock-runtime.us-west-2.amazonaws.com" {
			t.Fatalf("baseURL = %q", target.baseURL)
		}
		if target.service != "bedrock" || target.chatPath != "/openai/v1/chat/completions" || target.profile != "dev" {
			t.Fatalf("unexpected target: %#v", target)
		}
	})
	t.Run("mantle from attributes", func(t *testing.T) {
		target := bedrockTargetFromAuth(&cliproxyauth.Auth{Attributes: map[string]string{
			"bedrock_endpoint": "mantle",
			"aws_region":       "eu-west-1",
			"api_key":          "k",
			"base_url":         "https://bedrock-mantle.eu-west-1.api.aws/",
		}})
		if target.baseURL != "https://bedrock-mantle.eu-west-1.api.aws" {
			t.Fatalf("baseURL = %q", target.baseURL)
		}
		if target.service != "bedrock-mantle" || target.chatPath != "/v1/chat/completions" || target.apiKey != "k" {
			t.Fatalf("unexpected target: %#v", target)
		}
	})
}

func newBedrockTestAuth(baseURL string, extra map[string]string) *cliproxyauth.Auth {
	attrs := map[string]string{
		"source":           "config:bedrock[test]",
		"auth_kind":        "apikey",
		"base_url":         baseURL,
		"bedrock_endpoint": "runtime",
		"aws_region":       "us-east-1",
	}
	for k, v := range extra {
		attrs[k] = v
	}
	return &cliproxyauth.Auth{ID: "bedrock-test", Provider: "bedrock", Attributes: attrs}
}

func staticBedrockCreds(loads *int32) *helps.AWSCredentialsCache {
	return helps.NewAWSCredentialsCacheWithLoader(func(ctx context.Context, profile, region string) (aws.CredentialsProvider, error) {
		atomic.AddInt32(loads, 1)
		return credentials.NewStaticCredentialsProvider("AKIATEST", "secret", "token"), nil
	})
}

func TestBedrockExecutor_ExecuteMessagesWithAPIKey(t *testing.T) {
	var gotPath, gotAPIKey, gotVersion, gotAuthz, gotModel string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAPIKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotAuthz = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		gotModel = gjson.GetBytes(body, "model").String()
		if gjson.GetBytes(body, "stream").Bool() {
			t.Errorf("non-stream request should not set stream=true")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"global.anthropic.claude-sonnet-5","content":[{"type":"text","text":"hello from bedrock"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":4}}`))
	}))
	defer server.Close()

	exec := NewBedrockExecutor(&config.Config{})
	auth := newBedrockTestAuth(server.URL, map[string]string{"api_key": "bedrock-key"})
	payload := []byte(`{"model":"global.anthropic.claude-sonnet-5","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	resp, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "global.anthropic.claude-sonnet-5", Payload: payload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude})
	if err != nil {
		t.Fatalf("Execute error = %v", err)
	}
	if gotPath != "/anthropic/v1/messages" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAPIKey != "bedrock-key" || gotAuthz != "" {
		t.Fatalf("api key header = %q, authorization = %q", gotAPIKey, gotAuthz)
	}
	if gotVersion != bedrockAnthropicVersion {
		t.Fatalf("anthropic-version = %q", gotVersion)
	}
	if gotModel != "global.anthropic.claude-sonnet-5" {
		t.Fatalf("upstream model = %q", gotModel)
	}
	if !strings.Contains(string(resp.Payload), "hello from bedrock") {
		t.Fatalf("unexpected payload: %s", resp.Payload)
	}
}

func TestBedrockExecutor_ExecuteChatCompletionsWithSigV4(t *testing.T) {
	var gotPath, gotAuthz, gotDate, gotToken, gotAPIKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuthz = r.Header.Get("Authorization")
		gotDate = r.Header.Get("X-Amz-Date")
		gotToken = r.Header.Get("X-Amz-Security-Token")
		gotAPIKey = r.Header.Get("x-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"openai.gpt-oss-120b","choices":[{"index":0,"message":{"role":"assistant","content":"oss says hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}`))
	}))
	defer server.Close()

	var loads int32
	exec := NewBedrockExecutor(&config.Config{})
	exec.creds = staticBedrockCreds(&loads)
	auth := newBedrockTestAuth(server.URL, map[string]string{
		"bedrock_endpoint":              "mantle",
		"aws_region":                    "us-west-2",
		"aws_profile":                   "dev",
		"bedrock_chat_completions_path": "/v1/chat/completions",
	})
	payload := []byte(`{"model":"openai.gpt-oss-120b","messages":[{"role":"user","content":"hi"}]}`)
	resp, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "openai.gpt-oss-120b", Payload: payload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI})
	if err != nil {
		t.Fatalf("Execute error = %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q", gotPath)
	}
	if !strings.HasPrefix(gotAuthz, "AWS4-HMAC-SHA256 Credential=AKIATEST/") || !strings.Contains(gotAuthz, "/us-west-2/bedrock-mantle/aws4_request") {
		t.Fatalf("authorization = %q", gotAuthz)
	}
	if gotDate == "" || gotToken != "token" || gotAPIKey != "" {
		t.Fatalf("sigv4 headers: date=%q token=%q apikey=%q", gotDate, gotToken, gotAPIKey)
	}
	if atomic.LoadInt32(&loads) != 1 {
		t.Fatalf("credential loads = %d, want 1", loads)
	}
	if gjson.GetBytes(resp.Payload, "choices.0.message.content").String() != "oss says hi" {
		t.Fatalf("unexpected payload: %s", resp.Payload)
	}
}

func TestBedrockExecutor_ExpiredCredentialsInvalidateCache(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"The security token included in the request is expired"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer server.Close()

	var loads int32
	exec := NewBedrockExecutor(&config.Config{})
	exec.creds = staticBedrockCreds(&loads)
	auth := newBedrockTestAuth(server.URL, map[string]string{"aws_profile": "dev"})
	payload := []byte(`{"model":"openai.gpt-oss-120b-1:0","messages":[{"role":"user","content":"hi"}]}`)
	req := cliproxyexecutor.Request{Model: "openai.gpt-oss-120b-1:0", Payload: payload}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI}

	_, err := exec.Execute(context.Background(), auth, req, opts)
	var se statusErr
	if err == nil || !errorsAsStatusErr(err, &se) || se.code != http.StatusForbidden {
		t.Fatalf("expected 403 statusErr, got %v", err)
	}
	if _, err = exec.Execute(context.Background(), auth, req, opts); err != nil {
		t.Fatalf("second Execute error = %v", err)
	}
	if atomic.LoadInt32(&loads) != 2 {
		t.Fatalf("credential loads = %d, want 2 (cache invalidated after expiry)", loads)
	}
}

func errorsAsStatusErr(err error, target *statusErr) bool {
	se, ok := err.(statusErr)
	if !ok {
		return false
	}
	*target = se
	return true
}

func TestBedrockExecutor_ExecuteStreamMessagesPassthrough(t *testing.T) {
	events := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"global.anthropic.claude-sonnet-5\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
	var gotStream bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotStream = gjson.GetBytes(body, "stream").Bool()
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			_, _ = io.WriteString(w, ev)
		}
	}))
	defer server.Close()

	exec := NewBedrockExecutor(&config.Config{})
	auth := newBedrockTestAuth(server.URL, map[string]string{"api_key": "bedrock-key"})
	payload := []byte(`{"model":"global.anthropic.claude-sonnet-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	result, err := exec.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{Model: "global.anthropic.claude-sonnet-5", Payload: payload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude, Stream: true})
	if err != nil {
		t.Fatalf("ExecuteStream error = %v", err)
	}
	var chunks []string
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		chunks = append(chunks, string(chunk.Payload))
	}
	if !gotStream {
		t.Fatal("upstream request should set stream=true")
	}
	if len(chunks) != len(events) {
		t.Fatalf("chunk count = %d, want %d: %q", len(chunks), len(events), chunks)
	}
	if !strings.HasPrefix(chunks[0], "event: message_start") || !strings.Contains(chunks[2], "\"text\":\"hi\"") {
		t.Fatalf("unexpected chunks: %q", chunks)
	}
}

func TestBedrockExecutor_ExecuteStreamChatCompletions(t *testing.T) {
	var includeUsage bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		includeUsage = gjson.GetBytes(body, "stream_options.include_usage").Bool()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"openai.gpt-oss-120b-1:0\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, ": keep-alive\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"openai.gpt-oss-120b-1:0\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	var loads int32
	exec := NewBedrockExecutor(&config.Config{})
	exec.creds = staticBedrockCreds(&loads)
	auth := newBedrockTestAuth(server.URL, nil)
	payload := []byte(`{"model":"openai.gpt-oss-120b-1:0","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	result, err := exec.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{Model: "openai.gpt-oss-120b-1:0", Payload: payload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI, Stream: true})
	if err != nil {
		t.Fatalf("ExecuteStream error = %v", err)
	}
	var chunks []string
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		chunks = append(chunks, string(chunk.Payload))
	}
	if !includeUsage {
		t.Fatal("streaming chat request should request stream_options.include_usage")
	}
	// The translator emits bare JSON frames (the HTTP layer re-adds SSE framing)
	// and swallows the terminal [DONE] marker; comments must never leak through.
	if len(chunks) != 2 {
		t.Fatalf("chunk count = %d, want 2: %q", len(chunks), chunks)
	}
	if !strings.Contains(chunks[0], `"content":"hi"`) || gjson.Get(chunks[1], "usage.total_tokens").Int() != 3 {
		t.Fatalf("unexpected stream output: %q", chunks)
	}
	for _, chunk := range chunks {
		if strings.Contains(chunk, "keep-alive") {
			t.Fatalf("SSE comments must not be forwarded: %q", chunk)
		}
	}
}

func TestBedrockExecutor_UpstreamErrorCarriesRetryAfter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"message":"Too many requests"}`))
	}))
	defer server.Close()

	exec := NewBedrockExecutor(&config.Config{})
	auth := newBedrockTestAuth(server.URL, map[string]string{"api_key": "k"})
	payload := []byte(`{"model":"global.anthropic.claude-sonnet-5","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	_, err := exec.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{Model: "global.anthropic.claude-sonnet-5", Payload: payload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude, Stream: true})
	var se statusErr
	if err == nil || !errorsAsStatusErr(err, &se) {
		t.Fatalf("expected statusErr, got %v", err)
	}
	if se.code != http.StatusTooManyRequests || se.retryAfter == nil || se.retryAfter.Seconds() != 7 {
		t.Fatalf("unexpected statusErr: %#v", se)
	}
}

func TestBedrockExecutor_CountTokens(t *testing.T) {
	exec := NewBedrockExecutor(&config.Config{})
	auth := newBedrockTestAuth("https://bedrock-runtime.us-east-1.amazonaws.com", map[string]string{"api_key": "k"})

	claudePayload := []byte(`{"model":"global.anthropic.claude-sonnet-5","max_tokens":64,"messages":[{"role":"user","content":"count these tokens please"}]}`)
	resp, err := exec.CountTokens(context.Background(), auth, cliproxyexecutor.Request{Model: "global.anthropic.claude-sonnet-5", Payload: claudePayload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude})
	if err != nil {
		t.Fatalf("CountTokens (claude) error = %v", err)
	}
	if gjson.GetBytes(resp.Payload, "input_tokens").Int() <= 0 {
		t.Fatalf("expected positive claude token count, got %s", resp.Payload)
	}

	openaiPayload := []byte(`{"model":"openai.gpt-oss-120b-1:0","messages":[{"role":"user","content":"count these tokens please"}]}`)
	resp, err = exec.CountTokens(context.Background(), auth, cliproxyexecutor.Request{Model: "openai.gpt-oss-120b-1:0", Payload: openaiPayload}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI})
	if err != nil {
		t.Fatalf("CountTokens (openai) error = %v", err)
	}
	var decoded map[string]any
	if errJSON := json.Unmarshal(resp.Payload, &decoded); errJSON != nil {
		t.Fatalf("invalid token count payload %s: %v", resp.Payload, errJSON)
	}
	if fmt.Sprint(decoded) == "map[]" {
		t.Fatalf("empty token count payload")
	}
}
