package executor

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"net/http"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

// ExecuteStream performs a streaming request against Bedrock and translates the
// SSE stream into the client's response format.
func (e *BedrockExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	ctx = helps.EnsureSessionContext(ctx, opts, req.Payload)
	if opts.Alt == "responses/compact" {
		return nil, statusErr{code: http.StatusNotImplemented, msg: "/responses/compact not supported"}
	}
	if auth == nil {
		return nil, statusErr{code: http.StatusUnauthorized, msg: "bedrock executor: missing auth"}
	}

	target := bedrockTargetFromAuth(auth)
	prepared, errPrepare := e.prepareRequest(ctx, target, req, opts, true)
	if errPrepare != nil {
		return nil, errPrepare
	}

	reporter := helps.NewExecutorUsageReporter(ctx, e, prepared.baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)
	reporter.SetTranslatedReasoningEffort(prepared.body, prepared.to.String())

	httpReq, errRequest := e.buildRequest(ctx, auth, target, prepared, opts.Headers, true)
	if errRequest != nil {
		return nil, errRequest
	}
	e.recordRequest(ctx, auth, httpReq, prepared.body)

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, errDo := httpClient.Do(httpReq)
	if errDo != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errDo)
		return nil, errDo
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, errRead := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("bedrock executor: close response body error: %v", errClose)
		}
		if errRead != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errRead)
			return nil, errRead
		}
		helps.AppendAPIResponseChunk(ctx, e.cfg, data)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		return nil, e.wrapUpstreamError(target, httpResp.StatusCode, httpResp.Header, data)
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("bedrock executor: close response body error: %v", errClose)
			}
		}()
		var streamUsage helps.StreamUsageBuffer
		defer streamUsage.Publish(ctx, reporter)

		emit := func(chunk cliproxyexecutor.StreamChunk) bool {
			select {
			case out <- chunk:
				return true
			case <-ctx.Done():
				return false
			}
		}

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB
		switch prepared.protocol {
		case bedrockProtocolMessages:
			e.streamMessages(ctx, scanner, prepared, req, opts, &streamUsage, emit)
		case bedrockProtocolResponses:
			e.streamResponses(ctx, scanner, prepared, req, opts, reporter, emit)
		default:
			e.streamChatCompletions(ctx, scanner, prepared, req, opts, &streamUsage, emit)
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			streamUsage.PublishFailure(ctx, reporter, errScan)
			emit(cliproxyexecutor.StreamChunk{Err: errScan})
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

// streamMessages forwards an Anthropic Messages SSE stream. When the client also
// speaks the Claude format, whole SSE events are forwarded verbatim; otherwise
// each line is run through the translator like the native Claude executor does.
func (e *BedrockExecutor) streamMessages(ctx context.Context, scanner *bufio.Scanner, prepared *bedrockPreparedRequest, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, streamUsage *helps.StreamUsageBuffer, emit func(cliproxyexecutor.StreamChunk) bool) {
	passthrough := prepared.responseFormat == prepared.to
	var event bytes.Buffer
	var param any
	flushEvent := func() bool {
		if event.Len() == 0 {
			return true
		}
		cloned := bytes.Clone(event.Bytes())
		event.Reset()
		return emit(cliproxyexecutor.StreamChunk{Payload: cloned})
	}
	for scanner.Scan() {
		line := scanner.Bytes()
		helps.AppendAPIResponseChunk(ctx, e.cfg, line)
		streamUsage.ObserveClaudeStream(line)
		if passthrough {
			event.Write(line)
			event.WriteByte('\n')
			if len(bytes.TrimSpace(line)) == 0 && !flushEvent() {
				return
			}
			continue
		}
		chunks := sdktranslator.TranslateStream(ctx, prepared.to, prepared.responseFormat, req.Model, opts.OriginalRequest, prepared.body, bytes.Clone(line), &param)
		if prepared.responseFormat == sdktranslator.FormatOpenAIResponse {
			for i, chunk := range chunks {
				chunks[i] = helps.EnsureResponsesUsageDetails(chunk)
			}
		}
		for i := range chunks {
			if !emit(cliproxyexecutor.StreamChunk{Payload: chunks[i]}) {
				return
			}
		}
	}
	if passthrough {
		flushEvent()
	}
}

// streamChatCompletions forwards an OpenAI Chat Completions SSE stream, translating
// each data frame into the client's response format.
func (e *BedrockExecutor) streamChatCompletions(ctx context.Context, scanner *bufio.Scanner, prepared *bedrockPreparedRequest, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, streamUsage *helps.StreamUsageBuffer, emit func(cliproxyexecutor.StreamChunk) bool) {
	claudeInputTokens := helps.NewClaudeInputTokenState(prepared.from, prepared.to, prepared.responseFormat, prepared.originalPayload)
	var param any
	for scanner.Scan() {
		line := scanner.Bytes()
		helps.AppendAPIResponseChunk(ctx, e.cfg, line)
		streamUsage.ObserveOpenAIStream(line)
		trimmed := bytes.TrimSpace(line)
		if !bytes.HasPrefix(trimmed, dataTag) {
			continue
		}
		payload := bytes.TrimSpace(trimmed[len(dataTag):])
		streamLine := append([]byte("data: "), payload...)
		chunks := helps.TranslateStreamWithClaudeInputTokens(ctx, prepared.to, prepared.responseFormat, req.Model, opts.OriginalRequest, prepared.body, streamLine, &param, claudeInputTokens)
		for i := range chunks {
			if !emit(cliproxyexecutor.StreamChunk{Payload: chunks[i]}) {
				return
			}
		}
	}
}

// streamResponses forwards an OpenAI Responses API SSE stream, translating each data
// frame and publishing usage from the terminal response.completed event.
func (e *BedrockExecutor) streamResponses(ctx context.Context, scanner *bufio.Scanner, prepared *bedrockPreparedRequest, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, reporter *helps.UsageReporter, emit func(cliproxyexecutor.StreamChunk) bool) {
	claudeInputTokens := helps.NewClaudeInputTokenState(prepared.from, prepared.to, prepared.responseFormat, prepared.originalPayload)
	var param any
	outputItemsByIndex := make(map[int64][]byte)
	var outputItemsFallback [][]byte
	for scanner.Scan() {
		line := scanner.Bytes()
		helps.AppendAPIResponseChunk(ctx, e.cfg, line)
		trimmed := bytes.TrimSpace(line)
		if !bytes.HasPrefix(trimmed, dataTag) {
			continue
		}
		eventData := bytes.TrimSpace(trimmed[len(dataTag):])
		if errEvent := bedrockResponsesEventError(eventData); errEvent != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errEvent)
			reporter.PublishFailure(ctx, errEvent)
			emit(cliproxyexecutor.StreamChunk{Err: errEvent})
			return
		}
		switch gjson.GetBytes(eventData, "type").String() {
		case "response.output_item.done":
			xaiCollectOutputItemDone(eventData, outputItemsByIndex, &outputItemsFallback)
		case "response.completed", "response.incomplete":
			if detail, ok := helps.ParseCodexUsage(eventData); ok {
				reporter.Publish(ctx, detail)
			}
			eventData = patchCodexCompletedOutput(eventData, outputItemsByIndex, outputItemsFallback)
		}
		streamLine := append([]byte("data: "), eventData...)
		chunks := helps.TranslateStreamWithClaudeInputTokens(ctx, prepared.to, prepared.responseFormat, req.Model, opts.OriginalRequest, prepared.body, streamLine, &param, claudeInputTokens)
		for i := range chunks {
			if !emit(cliproxyexecutor.StreamChunk{Payload: chunks[i]}) {
				return
			}
		}
	}
}
