package management

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

func newBedrockTestHandler(t *testing.T) *Handler {
	t.Helper()
	cfg := &config.Config{BedrockKey: []config.BedrockKey{
		{Profile: "dev", Region: "us-west-2"},
		{APIKey: "bedrock-key", Endpoint: "mantle", Region: "us-east-1"},
	}}
	cfg.SanitizeBedrockKeys()
	return &Handler{cfg: cfg, configFilePath: writeTestConfigFile(t)}
}

func TestPutBedrockKeysNormalizesAndDerivesBaseURL(t *testing.T) {
	h := &Handler{cfg: &config.Config{}, configFilePath: writeTestConfigFile(t)}
	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodPut, "/v0/management/bedrock-api-key", strings.NewReader(`[
		{"profile": " dev ", "region": "eu-west-1", "models": [{"name": "global.anthropic.claude-sonnet-5", "alias": "claude-sonnet-5"}, {"name": " ", "alias": ""}]},
		{"api-key": "k", "endpoint": "bedrock-mantle", "region": "us-east-1"}
	]`))
	ctx.Request.Header.Set("Content-Type", "application/json")

	h.PutBedrockKeys(ctx)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if len(h.cfg.BedrockKey) != 2 {
		t.Fatalf("entries = %d, want 2", len(h.cfg.BedrockKey))
	}
	first := h.cfg.BedrockKey[0]
	if first.Profile != "dev" || first.BaseURL != "https://bedrock-runtime.eu-west-1.amazonaws.com" || len(first.Models) != 1 {
		t.Fatalf("first entry not normalized: %#v", first)
	}
	second := h.cfg.BedrockKey[1]
	if second.Endpoint != config.BedrockEndpointMantle || second.BaseURL != "https://bedrock-mantle.us-east-1.api.aws" {
		t.Fatalf("second entry not normalized: %#v", second)
	}
}

func TestPatchBedrockKeyUpdatesFields(t *testing.T) {
	h := newBedrockTestHandler(t)
	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodPatch, "/v0/management/bedrock-api-key", strings.NewReader(`{
		"index": 0,
		"value": {
			"region": "ca-central-1",
			"endpoint": "mantle",
			"base-url": "",
			"priority": 7,
			"disable-cooling": true,
			"request-retry": 0,
			"models": [{"name": "openai.gpt-oss-120b", "alias": "gpt-oss"}]
		}
	}`))
	ctx.Request.Header.Set("Content-Type", "application/json")

	h.PatchBedrockKey(ctx)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", rec.Code, http.StatusOK, rec.Body.String())
	}
	entry := h.cfg.BedrockKey[0]
	if entry.Region != "ca-central-1" || entry.Endpoint != config.BedrockEndpointMantle {
		t.Fatalf("region/endpoint not updated: %#v", entry)
	}
	if entry.BaseURL != "https://bedrock-mantle.ca-central-1.api.aws" {
		t.Fatalf("base-url should be re-derived after patch, got %q", entry.BaseURL)
	}
	if entry.Priority != 7 || entry.DisableCooling == nil || !*entry.DisableCooling || entry.RequestRetry == nil || *entry.RequestRetry != 0 {
		t.Fatalf("execution fields not updated: %#v", entry)
	}
	if len(entry.Models) != 1 || entry.Models[0].Alias != "gpt-oss" {
		t.Fatalf("models not updated: %#v", entry.Models)
	}
	if h.cfg.BedrockKey[1].APIKey != "bedrock-key" {
		t.Fatal("untouched entry was modified")
	}
}

func TestDeleteBedrockKeyByProfileAndIndex(t *testing.T) {
	h := newBedrockTestHandler(t)

	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodDelete, "/v0/management/bedrock-api-key?profile=dev", nil)
	h.DeleteBedrockKey(ctx)
	if rec.Code != http.StatusOK {
		t.Fatalf("delete by profile status = %d; body=%s", rec.Code, rec.Body.String())
	}
	if len(h.cfg.BedrockKey) != 1 || h.cfg.BedrockKey[0].APIKey != "bedrock-key" {
		t.Fatalf("unexpected remaining entries: %#v", h.cfg.BedrockKey)
	}

	rec = httptest.NewRecorder()
	ctx, _ = gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodDelete, "/v0/management/bedrock-api-key?index=0", nil)
	h.DeleteBedrockKey(ctx)
	if rec.Code != http.StatusOK || len(h.cfg.BedrockKey) != 0 {
		t.Fatalf("delete by index status = %d, remaining = %d", rec.Code, len(h.cfg.BedrockKey))
	}

	rec = httptest.NewRecorder()
	ctx, _ = gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodDelete, "/v0/management/bedrock-api-key", nil)
	h.DeleteBedrockKey(ctx)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("delete without selector status = %d, want 400", rec.Code)
	}
}

func TestBedrockKeysWithAuthIndexUsesStableID(t *testing.T) {
	h := newBedrockTestHandler(t)
	out := h.bedrockKeysWithAuthIndex()
	if len(out) != 2 {
		t.Fatalf("entries = %d, want 2", len(out))
	}
	if out[0].Profile != "dev" || out[1].APIKey != "bedrock-key" {
		t.Fatalf("unexpected entries: %#v", out)
	}
	a := bedrockAuthIDComponents(h.cfg.BedrockKey[0])
	b := bedrockAuthIDComponents(h.cfg.BedrockKey[1])
	if strings.Join(a, "|") == strings.Join(b, "|") {
		t.Fatal("distinct entries must produce distinct ID components")
	}
}
