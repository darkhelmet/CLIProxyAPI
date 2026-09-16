package executor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type bedrockPreparedRequest struct {
	baseModel       string
	protocol        bedrockProtocol
	messages        bool
	from            sdktranslator.Format
	responseFormat  sdktranslator.Format
	to              sdktranslator.Format
	originalPayload []byte
	body            []byte
	url             string
}

// prepareRequest translates the client payload into the upstream wire format for
// the selected Bedrock API and resolves the target URL.
func (e *BedrockExecutor) prepareRequest(ctx context.Context, target bedrockTarget, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, stream bool) (*bedrockPreparedRequest, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	protocol := bedrockProtocolFor(baseModel, target.openAIAPI)
	messages := protocol == bedrockProtocolMessages
	from := opts.SourceFormat
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	to := protocol.format()

	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := bytes.Clone(originalPayloadSource)
	isCompat := helps.APIKeyModelIsCompat(req)
	originalTranslated := helps.TranslateRequestWithAPIKeyModelCompatibility(ctx, opts.Headers, e.cfg, from, to, baseModel, originalPayload, stream, isCompat)
	body := helps.TranslateRequestWithAPIKeyModelCompatibility(ctx, opts.Headers, e.cfg, from, to, baseModel, bytes.Clone(req.Payload), stream, isCompat)

	var errThinking error
	body, errThinking = helps.ApplyRequestThinking(body, req, opts, from.String(), to.String(), e.Identifier())
	if errThinking != nil {
		return nil, errThinking
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = helps.ApplyPayloadConfigWithRequest(e.cfg, baseModel, to.String(), from.String(), "", body, originalTranslated, requestedModel, requestPath, opts.Headers)
	body = helps.SetStringIfDifferent(body, "model", baseModel)
	body = helps.SetBoolIfDifferent(body, "stream", stream)

	path := config.BedrockMessagesPath()
	switch protocol {
	case bedrockProtocolChatCompletions:
		path = target.chatPath
		body = renameBedrockMaxTokens(body)
		if stream {
			// Ask for usage in the final chunk so token accounting works.
			body = helps.SetBoolIfDifferent(body, "stream_options.include_usage", true)
		}
	case bedrockProtocolResponses:
		path = target.responsesPath
		body = prepareBedrockResponsesBody(ctx, body, target.endpoint)
	}

	return &bedrockPreparedRequest{
		baseModel:       baseModel,
		protocol:        protocol,
		messages:        messages,
		from:            from,
		responseFormat:  responseFormat,
		to:              to,
		originalPayload: originalPayload,
		body:            body,
		url:             target.baseURL + path,
	}, nil
}

// Execute performs a non-streaming request against Bedrock.
func (e *BedrockExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	ctx = helps.EnsureSessionContext(ctx, opts, req.Payload)
	if opts.Alt == "responses/compact" {
		return resp, statusErr{code: http.StatusNotImplemented, msg: "/responses/compact not supported"}
	}
	if auth == nil {
		return resp, statusErr{code: http.StatusUnauthorized, msg: "bedrock executor: missing auth"}
	}

	target := bedrockTargetFromAuth(auth)
	prepared, errPrepare := e.prepareRequest(ctx, target, req, opts, false)
	if errPrepare != nil {
		return resp, errPrepare
	}

	reporter := helps.NewExecutorUsageReporter(ctx, e, prepared.baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)
	reporter.SetTranslatedReasoningEffort(prepared.body, prepared.to.String())

	httpReq, errRequest := e.buildRequest(ctx, auth, target, prepared, opts.Headers, false)
	if errRequest != nil {
		return resp, errRequest
	}
	e.recordRequest(ctx, auth, httpReq, prepared.body)

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return resp, errDo
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("bedrock executor: close response body error: %v", errClose)
		}
	}()
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())

	data, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		return resp, errRead
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, data)
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		return resp, e.wrapUpstreamError(target, httpResp.StatusCode, httpResp.Header, data)
	}

	switch prepared.protocol {
	case bedrockProtocolMessages:
		reporter.Publish(ctx, helps.ParseClaudeUsage(data))
	case bedrockProtocolResponses:
		// The codex translators consume the response.completed event shape.
		completed, ok := metaAsCompletedEvent(data)
		if !ok {
			return resp, statusErr{code: http.StatusBadGateway, msg: "bedrock executor: unexpected responses payload"}
		}
		if errEvent := bedrockResponsesEventError(completed); errEvent != nil {
			return resp, errEvent
		}
		data = completed
		if detail, okUsage := helps.ParseCodexUsage(data); okUsage {
			reporter.Publish(ctx, detail)
		}
	default:
		reporter.Publish(ctx, helps.ParseOpenAIUsage(data))
	}
	reporter.EnsurePublished(ctx)

	var param any
	out := sdktranslator.TranslateNonStream(ctx, prepared.to, prepared.responseFormat, req.Model, opts.OriginalRequest, prepared.body, data, &param)
	if prepared.responseFormat == sdktranslator.FormatOpenAIResponse {
		out = helps.EnsureResponsesUsageDetails(out)
	}
	return cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}, nil
}

// CountTokens estimates input tokens locally; Bedrock token counting is not
// uniformly available across endpoints and models.
func (e *BedrockExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	if auth == nil {
		return cliproxyexecutor.Response{}, statusErr{code: http.StatusUnauthorized, msg: "bedrock executor: missing auth"}
	}
	target := bedrockTargetFromAuth(auth)
	prepared, errPrepare := e.prepareRequest(ctx, target, req, opts, false)
	if errPrepare != nil {
		return cliproxyexecutor.Response{}, errPrepare
	}
	if prepared.messages {
		count, errCount := helps.CountClaudeInputTokens(prepared.body)
		if errCount != nil {
			return cliproxyexecutor.Response{}, fmt.Errorf("bedrock executor: token counting failed: %w", errCount)
		}
		usageJSON := []byte(fmt.Sprintf(`{"input_tokens":%d}`, count))
		out := sdktranslator.TranslateTokenCount(ctx, prepared.to, prepared.responseFormat, count, usageJSON)
		return cliproxyexecutor.Response{Payload: out}, nil
	}
	enc, errEnc := helps.TokenizerForModel(prepared.baseModel)
	if errEnc != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("bedrock executor: tokenizer init failed: %w", errEnc)
	}
	if prepared.protocol == bedrockProtocolResponses {
		count, errCount := countCodexInputTokens(enc, prepared.body)
		if errCount != nil {
			return cliproxyexecutor.Response{}, fmt.Errorf("bedrock executor: token counting failed: %w", errCount)
		}
		usageJSON := fmt.Sprintf(`{"response":{"usage":{"input_tokens":%d,"output_tokens":0,"total_tokens":%d}}}`, count, count)
		out := sdktranslator.TranslateTokenCount(ctx, prepared.to, prepared.responseFormat, count, []byte(usageJSON))
		return cliproxyexecutor.Response{Payload: out}, nil
	}
	count, errCount := helps.CountOpenAIChatTokens(enc, prepared.body)
	if errCount != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("bedrock executor: token counting failed: %w", errCount)
	}
	out := sdktranslator.TranslateTokenCount(ctx, prepared.to, prepared.responseFormat, count, helps.BuildOpenAIUsageJSON(count))
	return cliproxyexecutor.Response{Payload: out}, nil
}

// renameBedrockMaxTokens rewrites the legacy max_tokens field to
// max_completion_tokens. Bedrock's OpenAI-compatible Chat Completions rejects
// max_tokens for GPT-5-class models ("Unsupported parameter: 'max_tokens'").
func renameBedrockMaxTokens(body []byte) []byte {
	maxTokens := gjson.GetBytes(body, "max_tokens")
	if !maxTokens.Exists() {
		return body
	}
	if !gjson.GetBytes(body, "max_completion_tokens").Exists() {
		if updated, err := sjson.SetRawBytes(body, "max_completion_tokens", []byte(maxTokens.Raw)); err == nil {
			body = updated
		}
	}
	if updated, err := sjson.DeleteBytes(body, "max_tokens"); err == nil {
		body = updated
	}
	return body
}

// prepareBedrockResponsesBody strips Codex-only fields that Bedrock's Responses API
// does not accept and normalizes reasoning replay content, mirroring the Meta executor.
func prepareBedrockResponsesBody(ctx context.Context, body []byte, endpoint string) []byte {
	for _, field := range []string{"generate", "prompt_cache_retention", "safety_identifier", "stream_options", "client_metadata", "prompt_cache_key"} {
		body, _ = sjson.DeleteBytes(body, field)
	}
	body = sanitizeBedrockResponsesTools(ctx, body, endpoint)
	body = normalizeCodexInstructions(body)
	return sanitizeOpenAIResponsesReasoningEncryptedContent(ctx, "bedrock executor", body)
}

// bedrockServerSideToolPrefixes lists Responses tool types executed by the provider
// rather than the client. bedrock-runtime does not offer server-side tools at all.
var bedrockServerSideToolPrefixes = []string{"web_search", "code_interpreter", "image_generation", "file_search", "mcp"}

// sanitizeBedrockResponsesTools adapts the Responses tools array to what Bedrock
// accepts: server-side tools are dropped on bedrock-runtime (unsupported there), and
// OpenAI-only web search options such as search_content_types are removed everywhere.
func sanitizeBedrockResponsesTools(ctx context.Context, body []byte, endpoint string) []byte {
	tools := gjson.GetBytes(body, "tools")
	if !tools.IsArray() {
		return body
	}
	runtime := config.NormalizeBedrockEndpoint(endpoint) == config.BedrockEndpointRuntime
	kept := make([]string, 0, len(tools.Array()))
	dropped := 0
	for _, tool := range tools.Array() {
		toolType := tool.Get("type").String()
		serverSide := false
		for _, prefix := range bedrockServerSideToolPrefixes {
			if strings.HasPrefix(toolType, prefix) {
				serverSide = true
				break
			}
		}
		if serverSide && runtime {
			dropped++
			continue
		}
		raw := tool.Raw
		if strings.HasPrefix(toolType, "web_search") {
			raw, _ = sjson.Delete(raw, "search_content_types")
		}
		kept = append(kept, raw)
	}
	if dropped > 0 {
		helps.LogWithRequestID(ctx).Debugf("bedrock executor: dropped %d server-side tool(s) unsupported on bedrock-runtime", dropped)
	}
	if len(kept) == 0 {
		body, _ = sjson.DeleteBytes(body, "tools")
		if choice := gjson.GetBytes(body, "tool_choice"); choice.Exists() && choice.String() != "auto" {
			body, _ = sjson.DeleteBytes(body, "tool_choice")
		}
		return body
	}
	body, _ = sjson.SetRawBytes(body, "tools", []byte("["+strings.Join(kept, ",")+"]"))
	return body
}

// bedrockResponsesEventError converts a Responses API error or failed event into a statusErr.
func bedrockResponsesEventError(eventData []byte) error {
	eventType := gjson.GetBytes(eventData, "type").String()
	if eventType != "error" && eventType != "response.failed" {
		return nil
	}
	statusCode := http.StatusBadGateway
	if code := int(gjson.GetBytes(eventData, "error.code").Int()); code >= 400 && code <= 599 {
		statusCode = code
	}
	return statusErr{code: statusCode, msg: string(eventData)}
}
