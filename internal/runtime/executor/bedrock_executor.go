package executor

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
)

var _ cliproxyauth.ProviderExecutor = (*BedrockExecutor)(nil)

const (
	bedrockAnthropicVersion = "2023-06-01"
	bedrockUserAgent        = "cli-proxy-bedrock"
)

// BedrockExecutor implements cliproxyauth.ProviderExecutor for AWS Bedrock.
//
// Anthropic models are sent to the Anthropic Messages API and every other model
// to the OpenAI Chat Completions API. Both bedrock-runtime and bedrock-mantle
// expose these routes over plain SSE. Requests are authenticated with a Bedrock
// API key (bearer) when configured, otherwise signed with AWS SigV4 using the
// credentials resolved from the configured AWS profile.
type BedrockExecutor struct {
	cfg   *config.Config
	creds *helps.AWSCredentialsCache
}

// NewBedrockExecutor constructs a new Bedrock executor.
func NewBedrockExecutor(cfg *config.Config) *BedrockExecutor {
	return &BedrockExecutor{cfg: cfg, creds: helps.NewAWSCredentialsCache()}
}

// Identifier returns the provider identifier "bedrock".
func (e *BedrockExecutor) Identifier() string {
	return "bedrock"
}

// bedrockTarget captures the resolved upstream endpoint and credential source for one auth.
type bedrockTarget struct {
	baseURL       string
	endpoint      string
	region        string
	service       string
	chatPath      string
	responsesPath string
	openAIAPI     string
	apiKey        string
	profile       string
}

// bedrockProtocol identifies the upstream wire API used for a model.
type bedrockProtocol int

const (
	bedrockProtocolMessages bedrockProtocol = iota
	bedrockProtocolChatCompletions
	bedrockProtocolResponses
)

func bedrockTargetFromAuth(a *cliproxyauth.Auth) bedrockTarget {
	var attrs map[string]string
	if a != nil {
		attrs = a.Attributes
	}
	get := func(key string) string {
		if attrs == nil {
			return ""
		}
		return strings.TrimSpace(attrs[key])
	}
	profile := get("aws_profile")
	endpoint := config.NormalizeBedrockEndpoint(get("bedrock_endpoint"))
	region := get("aws_region")
	if region == "" {
		region = config.ResolveBedrockRegion("", profile)
	}
	baseURL := strings.TrimRight(get("base_url"), "/")
	if baseURL == "" {
		baseURL = config.BedrockBaseURL(endpoint, region)
	}
	chatPath := get("bedrock_chat_completions_path")
	if chatPath == "" {
		chatPath = config.BedrockChatCompletionsPath(endpoint, "")
	}
	responsesPath := get("bedrock_responses_path")
	if responsesPath == "" {
		responsesPath = config.BedrockResponsesPath("")
	}
	return bedrockTarget{
		baseURL:       baseURL,
		endpoint:      endpoint,
		region:        region,
		service:       config.BedrockSigningService(endpoint),
		chatPath:      chatPath,
		responsesPath: responsesPath,
		openAIAPI:     config.NormalizeBedrockOpenAIAPI(get("bedrock_openai_api")),
		apiKey:        get("api_key"),
		profile:       profile,
	}
}

// bedrockProtocolFor picks the upstream API for a model. Anthropic models always use
// the Messages API. Other models follow the credential's openai-api setting; in auto
// mode GPT-5-class models use the Responses API (Bedrock rejects function tools with
// reasoning on Chat Completions for them) while gpt-oss stays on Chat Completions,
// which is the only OpenAI-compatible API it supports on bedrock-runtime.
func bedrockProtocolFor(model, openAIAPI string) bedrockProtocol {
	if bedrockUsesMessagesAPI(model) {
		return bedrockProtocolMessages
	}
	switch config.NormalizeBedrockOpenAIAPI(openAIAPI) {
	case config.BedrockOpenAIAPIResponses:
		return bedrockProtocolResponses
	case config.BedrockOpenAIAPIChatCompletions:
		return bedrockProtocolChatCompletions
	}
	if strings.Contains(strings.ToLower(model), "gpt-oss") {
		return bedrockProtocolChatCompletions
	}
	return bedrockProtocolResponses
}

func (p bedrockProtocol) format() sdktranslator.Format {
	switch p {
	case bedrockProtocolMessages:
		return sdktranslator.FormatClaude
	case bedrockProtocolResponses:
		return sdktranslator.FormatCodex
	default:
		return sdktranslator.FormatOpenAI
	}
}

// bedrockUsesMessagesAPI reports whether the upstream model is served by the
// Anthropic Messages API (true) or the OpenAI Chat Completions API (false).
func bedrockUsesMessagesAPI(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(m, "anthropic.") || strings.HasPrefix(m, "claude")
}

// RequestToFormat tells the conductor which wire format this request will use upstream.
// The credential is not available here, so the auto policy is assumed.
func (e *BedrockExecutor) RequestToFormat(req cliproxyexecutor.Request, _ cliproxyexecutor.Options) sdktranslator.Format {
	return bedrockProtocolFor(thinking.ParseSuffix(req.Model).ModelName, config.BedrockOpenAIAPIAuto).format()
}

// PrepareRequest applies Bedrock authentication to an outgoing HTTP request.
// Custom headers must already be present because SigV4 signs them.
func (e *BedrockExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	messages := req.URL != nil && strings.Contains(req.URL.Path, "/anthropic/")
	return e.applyAuth(req.Context(), req, bedrockTargetFromAuth(auth), messages)
}

// applyAuth sets bearer or SigV4 authentication headers on req.
func (e *BedrockExecutor) applyAuth(ctx context.Context, req *http.Request, target bedrockTarget, messages bool) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if target.apiKey != "" {
		if messages {
			req.Header.Set("x-api-key", target.apiKey)
			req.Header.Del("Authorization")
		} else {
			req.Header.Set("Authorization", "Bearer "+target.apiKey)
			req.Header.Del("x-api-key")
		}
		return nil
	}
	if e.creds == nil {
		e.creds = helps.NewAWSCredentialsCache()
	}
	creds, err := e.creds.Retrieve(ctx, target.profile, target.region)
	if err != nil {
		return statusErr{code: http.StatusUnauthorized, msg: fmt.Sprintf("bedrock executor: %v", err)}
	}
	req.Header.Del("x-api-key")
	if errSign := helps.SignAWSRequest(ctx, req, creds, target.service, target.region); errSign != nil {
		return statusErr{code: http.StatusInternalServerError, msg: fmt.Sprintf("bedrock executor: %v", errSign)}
	}
	return nil
}

// HttpRequest authenticates the request and executes it with the per-auth HTTP client.
func (e *BedrockExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("bedrock executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Refresh is a no-op for Bedrock: SigV4 credentials are refreshed by the AWS
// credential cache and API keys are static.
func (e *BedrockExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("bedrock executor: refresh called")
	if refreshed, handled, err := helps.RefreshAuthViaHome(ctx, e.cfg, auth); handled {
		return refreshed, err
	}
	if auth == nil {
		return nil, statusErr{code: http.StatusInternalServerError, msg: "bedrock executor: auth is nil"}
	}
	return auth, nil
}

// buildRequest assembles the upstream HTTP request with content and auth headers.
func (e *BedrockExecutor) buildRequest(ctx context.Context, auth *cliproxyauth.Auth, target bedrockTarget, prepared *bedrockPreparedRequest, clientHeaders http.Header, stream bool) (*http.Request, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, prepared.url, strings.NewReader(string(prepared.body)))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", bedrockUserAgent)
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	if prepared.messages {
		httpReq.Header.Set("anthropic-version", bedrockAnthropicVersion)
		if beta := strings.TrimSpace(clientHeaders.Get("anthropic-beta")); beta != "" {
			httpReq.Header.Set("anthropic-beta", beta)
		}
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs, clientHeaders)
	if errAuth := e.applyAuth(ctx, httpReq, target, prepared.messages); errAuth != nil {
		return nil, errAuth
	}
	return httpReq, nil
}

func (e *BedrockExecutor) recordRequest(ctx context.Context, auth *cliproxyauth.Auth, httpReq *http.Request, body []byte) {
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       httpReq.URL.String(),
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
}

// wrapUpstreamError converts a non-2xx Bedrock response into a statusErr, honoring
// Retry-After on throttling and dropping cached SigV4 credentials when AWS reports
// them as expired so the next attempt reloads the profile.
func (e *BedrockExecutor) wrapUpstreamError(target bedrockTarget, statusCode int, headers http.Header, body []byte) error {
	se := statusErr{code: statusCode, msg: string(body)}
	if statusCode == http.StatusTooManyRequests {
		if seconds, errParse := strconv.Atoi(strings.TrimSpace(headers.Get("Retry-After"))); errParse == nil && seconds > 0 {
			retryAfter := time.Duration(seconds) * time.Second
			se.retryAfter = &retryAfter
		}
	}
	if statusCode == http.StatusForbidden && target.apiKey == "" && e.creds != nil {
		lower := strings.ToLower(string(body))
		if strings.Contains(lower, "expired") || strings.Contains(lower, "invalidsignature") {
			e.creds.Invalidate(target.profile, target.region)
		}
	}
	return se
}
