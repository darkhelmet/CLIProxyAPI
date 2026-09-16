package executor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

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
	messages := bedrockUsesMessagesAPI(baseModel)
	from := opts.SourceFormat
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	to := bedrockUpstreamFormat(baseModel)

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
	if !messages {
		path = target.chatPath
		body = renameBedrockMaxTokens(body)
		if stream {
			// Ask for usage in the final chunk so token accounting works.
			body = helps.SetBoolIfDifferent(body, "stream_options.include_usage", true)
		}
	}

	return &bedrockPreparedRequest{
		baseModel:       baseModel,
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

	if prepared.messages {
		reporter.Publish(ctx, helps.ParseClaudeUsage(data))
	} else {
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
