package config

import "testing"

func TestBedrockConfigDerivesDefaults(t *testing.T) {
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/aws-credentials")

	cfg, err := ParseConfigBytes([]byte(`bedrock-api-key:
  - profile: " dev "
    region: "us-west-2"
    models:
      - name: "global.anthropic.claude-sonnet-5"
        alias: "claude-sonnet-5"
  - api-key: " bedrock-key "
    endpoint: "mantle"
  - profile: "prod"
    base-url: "https://vpce.example.com/"
    chat-completions-path: "openai/v1/chat/completions"
`))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.BedrockKey) != 3 {
		t.Fatalf("got %d keys, want 3", len(cfg.BedrockKey))
	}
	first := cfg.BedrockKey[0]
	if first.Profile != "dev" || first.Region != "us-west-2" || first.Endpoint != BedrockEndpointRuntime {
		t.Fatalf("first entry not normalized: %#v", first)
	}
	if first.BaseURL != "https://bedrock-runtime.us-west-2.amazonaws.com" {
		t.Fatalf("first base-url = %q", first.BaseURL)
	}
	second := cfg.BedrockKey[1]
	if second.APIKey != "bedrock-key" || second.Endpoint != BedrockEndpointMantle {
		t.Fatalf("second entry not normalized: %#v", second)
	}
	if second.Region != BedrockDefaultRegion {
		t.Fatalf("second region = %q, want default %q", second.Region, BedrockDefaultRegion)
	}
	if second.BaseURL != "https://bedrock-mantle.us-east-1.api.aws" {
		t.Fatalf("second base-url = %q", second.BaseURL)
	}
	third := cfg.BedrockKey[2]
	if third.BaseURL != "https://vpce.example.com" {
		t.Fatalf("third base-url = %q, want trailing slash trimmed", third.BaseURL)
	}
	if got := BedrockChatCompletionsPath(third.Endpoint, third.ChatCompletionsPath); got != "/openai/v1/chat/completions" {
		t.Fatalf("chat path override = %q", got)
	}
}

func TestBedrockPathsAndSigningService(t *testing.T) {
	if got := BedrockChatCompletionsPath(BedrockEndpointRuntime, ""); got != "/openai/v1/chat/completions" {
		t.Fatalf("runtime chat path = %q", got)
	}
	if got := BedrockChatCompletionsPath(BedrockEndpointMantle, ""); got != "/v1/chat/completions" {
		t.Fatalf("mantle chat path = %q", got)
	}
	if got := BedrockSigningService(BedrockEndpointRuntime); got != "bedrock" {
		t.Fatalf("runtime signing service = %q", got)
	}
	if got := BedrockSigningService("bedrock-mantle"); got != "bedrock-mantle" {
		t.Fatalf("mantle signing service = %q", got)
	}
	if got := BedrockMessagesPath(); got != "/anthropic/v1/messages" {
		t.Fatalf("messages path = %q", got)
	}
}

func TestBedrockRegionFromEnv(t *testing.T) {
	t.Setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/aws-credentials")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_REGION", "")
	t.Setenv("AWS_DEFAULT_REGION", "eu-west-1")
	if got := ResolveBedrockRegion("", ""); got != "eu-west-1" {
		t.Fatalf("region = %q, want eu-west-1", got)
	}
	if got := ResolveBedrockRegion(" ap-southeast-2 ", ""); got != "ap-southeast-2" {
		t.Fatalf("explicit region = %q", got)
	}
}
