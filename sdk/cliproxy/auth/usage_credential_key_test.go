package auth

import "testing"

func TestUsageCredentialKey(t *testing.T) {
	keyed := &Auth{Provider: "bedrock", Attributes: map[string]string{"api_key": "bedrock-key", "aws_profile": "dev"}}
	if got := keyed.UsageCredentialKey(); got != "bedrock-key" {
		t.Fatalf("api key should win, got %q", got)
	}
	profile := &Auth{Provider: "bedrock", Attributes: map[string]string{"auth_kind": "apikey", "aws_profile": "dev"}}
	if got := profile.UsageCredentialKey(); got != "dev" {
		t.Fatalf("profile key = %q, want dev", got)
	}
	chain := &Auth{Provider: "bedrock", Attributes: map[string]string{"auth_kind": "apikey"}}
	if got := chain.UsageCredentialKey(); got != "default" {
		t.Fatalf("default chain key = %q, want default", got)
	}
	other := &Auth{Provider: "claude", Attributes: map[string]string{}}
	if got := other.UsageCredentialKey(); got != "" {
		t.Fatalf("non-bedrock auth without key should be empty, got %q", got)
	}
	if (*Auth)(nil).UsageCredentialKey() != "" {
		t.Fatal("nil auth should be empty")
	}
}
