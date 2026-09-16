package config

import (
	"context"
	"os"
	"strings"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

// BedrockDefaultRegion is used when no region can be resolved from config, profile, or environment.
const BedrockDefaultRegion = "us-east-1"

// SanitizeBedrockKeys normalizes Bedrock credential entries and fills derived defaults
// (endpoint, region, base URL) so downstream matching on base_url is stable.
func (cfg *Config) SanitizeBedrockKeys() {
	if cfg == nil {
		return
	}
	cfg.BedrockKey = sanitizeBedrockKeyEntries(cfg.BedrockKey)
}

func sanitizeBedrockKeyEntries(entries []BedrockKey) []BedrockKey {
	if len(entries) == 0 {
		return entries
	}
	out := make([]BedrockKey, 0, len(entries))
	for i := range entries {
		e := entries[i]
		e.APIKey = strings.TrimSpace(e.APIKey)
		e.Profile = strings.TrimSpace(e.Profile)
		e.Prefix = normalizeModelPrefix(e.Prefix)
		e.ProxyURL = strings.TrimSpace(e.ProxyURL)
		e.Endpoint = NormalizeBedrockEndpoint(e.Endpoint)
		e.Region = ResolveBedrockRegion(e.Region, e.Profile)
		e.BaseURL = strings.TrimRight(strings.TrimSpace(e.BaseURL), "/")
		if e.BaseURL == "" {
			e.BaseURL = BedrockBaseURL(e.Endpoint, e.Region)
		}
		e.ChatCompletionsPath = strings.TrimSpace(e.ChatCompletionsPath)
		e.Headers = NormalizeHeaders(e.Headers)
		e.ExcludedModels = NormalizeExcludedModels(e.ExcludedModels)
		out = append(out, e)
	}
	return out
}

// NormalizeBedrockEndpoint returns "runtime" or "mantle", defaulting to runtime.
func NormalizeBedrockEndpoint(endpoint string) string {
	switch strings.ToLower(strings.TrimSpace(endpoint)) {
	case BedrockEndpointMantle, "bedrock-mantle":
		return BedrockEndpointMantle
	default:
		return BedrockEndpointRuntime
	}
}

// BedrockBaseURL derives the endpoint base URL for the given endpoint kind and region.
func BedrockBaseURL(endpoint, region string) string {
	region = strings.TrimSpace(region)
	if region == "" {
		region = BedrockDefaultRegion
	}
	if NormalizeBedrockEndpoint(endpoint) == BedrockEndpointMantle {
		return "https://bedrock-mantle." + region + ".api.aws"
	}
	return "https://bedrock-runtime." + region + ".amazonaws.com"
}

// BedrockSigningService returns the SigV4 service name for the endpoint kind.
func BedrockSigningService(endpoint string) string {
	if NormalizeBedrockEndpoint(endpoint) == BedrockEndpointMantle {
		return "bedrock-mantle"
	}
	return "bedrock"
}

// BedrockMessagesPath returns the Anthropic Messages API path (same on both endpoints).
func BedrockMessagesPath() string {
	return "/anthropic/v1/messages"
}

// BedrockChatCompletionsPath returns the OpenAI Chat Completions path for the endpoint kind,
// honoring an explicit override when provided.
func BedrockChatCompletionsPath(endpoint, override string) string {
	if p := strings.TrimSpace(override); p != "" {
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		return p
	}
	if NormalizeBedrockEndpoint(endpoint) == BedrockEndpointMantle {
		return "/v1/chat/completions"
	}
	return "/openai/v1/chat/completions"
}

// ResolveBedrockRegion picks the effective region: explicit config, then the AWS shared
// config profile, then AWS_REGION / AWS_DEFAULT_REGION, then the Bedrock default.
// Only local files and environment variables are consulted; no network calls are made.
func ResolveBedrockRegion(region, profile string) string {
	if r := strings.TrimSpace(region); r != "" {
		return r
	}
	if r := bedrockRegionFromSharedConfig(profile); r != "" {
		return r
	}
	for _, key := range []string{"AWS_REGION", "AWS_DEFAULT_REGION"} {
		if r := strings.TrimSpace(os.Getenv(key)); r != "" {
			return r
		}
	}
	return BedrockDefaultRegion
}

func bedrockRegionFromSharedConfig(profile string) string {
	profile = strings.TrimSpace(profile)
	if profile == "" {
		profile = strings.TrimSpace(os.Getenv("AWS_PROFILE"))
	}
	if profile == "" {
		profile = awsconfig.DefaultSharedConfigProfile
	}
	shared, err := awsconfig.LoadSharedConfigProfile(context.Background(), profile, func(o *awsconfig.LoadSharedConfigOptions) {
		// Honor the standard AWS environment overrides for file locations.
		if p := strings.TrimSpace(os.Getenv("AWS_CONFIG_FILE")); p != "" {
			o.ConfigFiles = []string{p}
		}
		if p := strings.TrimSpace(os.Getenv("AWS_SHARED_CREDENTIALS_FILE")); p != "" {
			o.CredentialsFiles = []string{p}
		}
	})
	if err != nil {
		return ""
	}
	return strings.TrimSpace(shared.Region)
}
