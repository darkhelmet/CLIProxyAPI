package cliproxy

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
)

func TestOpenAICompatibilityRegistrationCacheUsesConfigIndex(t *testing.T) {
	service := &Service{cfg: &config.Config{OpenAICompatibility: []config.OpenAICompatibility{
		{Name: "shared", Models: []config.OpenAICompatibilityModel{{Name: "first"}}},
		{Name: "shared", Models: []config.OpenAICompatibilityModel{{Name: "second"}}},
	}}}
	cache := service.newOpenAICompatibilityRegistrationCache()
	auth := &coreauth.Auth{Attributes: map[string]string{
		coreauth.AttributeSource:      "config:shared[token-1]",
		coreauth.AttributeConfigIndex: "1",
	}}
	entry, ok := cache.lookup(auth, "shared")
	if !ok || entry == nil || len(entry.models) != 1 || entry.models[0].ID != "second" {
		t.Fatalf("cached config entry = %+v, want second model", entry)
	}
}

func TestRegisterBedrockModelsUsesCapturedConfig(t *testing.T) {
	makeConfig := func(alias, excluded string, forcePrefix bool) *config.Config {
		return &config.Config{
			SDKConfig: config.SDKConfig{ForceModelPrefix: forcePrefix},
			BedrockKey: []config.BedrockKey{
				{BaseURL: "https://bedrock.example", Models: []config.BedrockModel{{Name: "wrong-entry"}}},
				{
					BaseURL: "https://bedrock.example",
					Models: []config.BedrockModel{
						{Name: "model", Alias: alias},
						{Name: "excluded-old"},
						{Name: "excluded-new"},
					},
					ExcludedModels: []string{excluded},
				},
			},
		}
	}
	captured := makeConfig("alias-old", "excluded-old", true)
	service := &Service{cfg: captured}
	service.commitConfigUpdate(makeConfig("alias-new", "excluded-new", false))
	auth := &coreauth.Auth{
		ID: "bedrock-config-snapshot", Provider: "bedrock", Prefix: "pfx",
		Attributes: map[string]string{
			"auth_kind": "apikey", "base_url": "https://bedrock.example",
			coreauth.AttributeSource: "config:bedrock[test]", coreauth.AttributeConfigIndex: "1",
		},
	}
	reg := GlobalModelRegistry()
	t.Cleanup(func() { reg.UnregisterClient(auth.ID) })
	service.registerModelsForAuthWithConfigAndCache(context.Background(), captured, auth, nil)
	for model, want := range map[string]bool{
		"pfx/alias-old": true, "pfx/excluded-new": true,
		"alias-old": false, "pfx/alias-new": false, "alias-new": false,
		"pfx/excluded-old": false, "pfx/wrong-entry": false,
	} {
		if got := reg.ClientSupportsModel(auth.ID, model); got != want {
			t.Errorf("ClientSupportsModel(%q) = %v, want %v from captured config", model, got, want)
		}
	}
}

func TestResolveConfigClaudeKeyUsesConfigIndex(t *testing.T) {
	service := &Service{cfg: &config.Config{ClaudeKey: []config.ClaudeKey{
		{APIKey: "shared-key", Models: []config.ClaudeModel{{Name: "first"}}},
		{APIKey: "shared-key", Models: []config.ClaudeModel{{Name: "second"}}},
	}}}
	auth := &coreauth.Auth{Attributes: map[string]string{
		coreauth.AttributeAPIKey:      "shared-key",
		coreauth.AttributeSource:      "config:claude[token-1]",
		coreauth.AttributeConfigIndex: "1",
	}}

	entry := service.resolveConfigClaudeKey(auth)
	if entry == nil || len(entry.Models) != 1 || entry.Models[0].Name != "second" {
		t.Fatalf("resolved config entry = %+v, want second entry", entry)
	}
}
