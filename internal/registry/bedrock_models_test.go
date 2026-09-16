package registry

import "testing"

func TestNormalizeBedrockModelID(t *testing.T) {
	cases := map[string]string{
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0":     "claude-sonnet-4-5-20250929",
		"global.anthropic.claude-sonnet-4-5-20250929-v1:0": "claude-sonnet-4-5-20250929",
		"anthropic.claude-sonnet-4-5-20250929-v1:0":        "claude-sonnet-4-5-20250929",
		"global.anthropic.claude-sonnet-5":                 "claude-sonnet-5",
		"anthropic.claude-sonnet-5":                        "claude-sonnet-5",
		"openai.gpt-oss-120b-1:0":                          "openai.gpt-oss-120b-1:0",
		"claude-sonnet-5":                                  "claude-sonnet-5",
	}
	for in, want := range cases {
		if got := NormalizeBedrockModelID(in); got != want {
			t.Errorf("NormalizeBedrockModelID(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBedrockStaticModelsAndVendorFallback(t *testing.T) {
	models := GetBedrockModels()
	if len(models) == 0 {
		t.Fatal("expected static bedrock models")
	}
	for _, m := range models {
		if m.Type != "bedrock" {
			t.Fatalf("model %s type = %q, want bedrock", m.ID, m.Type)
		}
	}
	if info := LookupStaticModelInfoByChannel("global.anthropic.claude-sonnet-5", "bedrock"); info == nil || info.Type != "bedrock" {
		t.Fatalf("expected direct bedrock lookup, got %#v", info)
	}

	// Not in the bedrock section, but resolvable through the Claude section.
	const opusID = "eu.anthropic.claude-opus-4-6"
	if direct := findStaticModel(GetBedrockModels(), opusID); direct != nil {
		t.Fatalf("test assumes %s is not a static bedrock model", opusID)
	}
	claude := LookupStaticModelInfo("claude-opus-4-6")
	if claude == nil {
		t.Fatal("expected claude-opus-4-6 in claude static models")
	}
	fallback := LookupStaticModelInfoByChannel(opusID, "bedrock")
	if fallback == nil {
		t.Fatal("expected bedrock channel lookup to fall back to claude static info")
	}
	if fallback.ID != opusID || fallback.Type != "bedrock" {
		t.Fatalf("fallback identity = (%q, %q)", fallback.ID, fallback.Type)
	}
	if fallback.ContextLength != claude.ContextLength || (claude.Thinking != nil && fallback.Thinking == nil) {
		t.Fatalf("fallback did not inherit claude capabilities: %#v", fallback)
	}
	if global := LookupStaticModelInfo(opusID); global == nil || global.ID != opusID {
		t.Fatalf("global lookup fallback = %#v", global)
	}
	if LookupStaticModelInfoByChannel("openai.gpt-oss-99b", "bedrock") != nil {
		t.Fatal("unknown non-anthropic bedrock model should not resolve")
	}
	if LookupStaticModelInfoByChannel(opusID, "claude") != nil {
		t.Fatal("claude channel must not accept bedrock IDs")
	}
}

func findStaticModel(models []*ModelInfo, id string) *ModelInfo {
	for _, m := range models {
		if m != nil && m.ID == id {
			return m
		}
	}
	return nil
}
