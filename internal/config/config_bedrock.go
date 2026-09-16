package config

import (
	"context"
	"os"
	"regexp"
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
		// A previously derived URL is not a user override: re-derive it so changing
		// the endpoint or region takes effect instead of pinning the old host.
		if e.BaseURL == "" || IsDerivedBedrockBaseURL(e.BaseURL) {
			e.BaseURL = BedrockBaseURL(e.Endpoint, e.Region)
		}
		e.ChatCompletionsPath = strings.TrimSpace(e.ChatCompletionsPath)
		e.ResponsesPath = strings.TrimSpace(e.ResponsesPath)
		e.OpenAIAPI = NormalizeBedrockOpenAIAPI(e.OpenAIAPI)
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

// IsDerivedBedrockBaseURL reports whether baseURL matches the standard AWS Bedrock
// endpoint shape produced by BedrockBaseURL (for any endpoint kind or region), as
// opposed to a custom override such as a VPC endpoint.
func IsDerivedBedrockBaseURL(baseURL string) bool {
	return bedrockDerivedBaseURLPattern.MatchString(strings.TrimRight(strings.TrimSpace(baseURL), "/"))
}

var bedrockDerivedBaseURLPattern = regexp.MustCompile(`^https://(bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com|bedrock-mantle\.[a-z0-9-]+\.api\.aws)$`)

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

// Bedrock OpenAI API selection values.
const (
	BedrockOpenAIAPIAuto            = "auto"
	BedrockOpenAIAPIResponses       = "responses"
	BedrockOpenAIAPIChatCompletions = "chat-completions"
)

// NormalizeBedrockOpenAIAPI returns "auto", "responses", or "chat-completions".
func NormalizeBedrockOpenAIAPI(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case BedrockOpenAIAPIResponses:
		return BedrockOpenAIAPIResponses
	case BedrockOpenAIAPIChatCompletions, "chat", "chat_completions":
		return BedrockOpenAIAPIChatCompletions
	default:
		return BedrockOpenAIAPIAuto
	}
}

// BedrockResponsesPath returns the OpenAI Responses API path, honoring an override.
func BedrockResponsesPath(override string) string {
	if p := strings.TrimSpace(override); p != "" {
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		return p
	}
	return "/openai/v1/responses"
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
