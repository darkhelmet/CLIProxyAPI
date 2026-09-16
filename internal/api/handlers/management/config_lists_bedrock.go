package management

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

// bedrock-api-key: []BedrockKey
func (h *Handler) GetBedrockKeys(c *gin.Context) {
	c.JSON(200, gin.H{"bedrock-api-key": h.bedrockKeysWithAuthIndex()})
}

func (h *Handler) PutBedrockKeys(c *gin.Context) {
	data, errRead := c.GetRawData()
	if errRead != nil {
		c.JSON(400, gin.H{"error": "failed to read body"})
		return
	}
	var arr []config.BedrockKey
	if errUnmarshal := json.Unmarshal(data, &arr); errUnmarshal != nil {
		var obj struct {
			Items []config.BedrockKey `json:"items"`
		}
		if errObject := json.Unmarshal(data, &obj); errObject != nil || len(obj.Items) == 0 {
			c.JSON(400, gin.H{"error": "invalid body"})
			return
		}
		arr = obj.Items
	}
	filtered := make([]config.BedrockKey, 0, len(arr))
	for i := range arr {
		entry := arr[i]
		normalizeBedrockKey(&entry)
		if rejectInvalidCredentialWeight(c, fmt.Sprintf("bedrock-api-key[%d].weight", i), entry.Weight) {
			return
		}
		filtered = append(filtered, entry)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.cfg.BedrockKey = filtered
	h.cfg.SanitizeBedrockKeys()
	h.persistLocked(c)
}

func (h *Handler) PatchBedrockKey(c *gin.Context) {
	type bedrockKeyPatch struct {
		APIKey              *string                          `json:"api-key"`
		Profile             *string                          `json:"profile"`
		Region              *string                          `json:"region"`
		Endpoint            *string                          `json:"endpoint"`
		BaseURL             *string                          `json:"base-url"`
		ChatCompletionsPath *string                          `json:"chat-completions-path"`
		Priority            *int                             `json:"priority"`
		Weight              json.RawMessage                  `json:"weight"`
		Prefix              *string                          `json:"prefix"`
		ProxyURL            *string                          `json:"proxy-url"`
		Models              *[]config.BedrockModel           `json:"models"`
		Headers             *map[string]string               `json:"headers"`
		ExcludedModels      *[]string                        `json:"excluded-models"`
		DisableCooling      json.RawMessage                  `json:"disable-cooling"`
		RequestRetry        *int                             `json:"request-retry"`
		RequestScopedErrors *[]config.RequestScopedErrorRule `json:"request-scoped-errors"`
	}
	var body struct {
		Index *int             `json:"index"`
		Value *bedrockKeyPatch `json:"value"`
	}
	if errBind := c.ShouldBindJSON(&body); errBind != nil || body.Value == nil {
		c.JSON(400, gin.H{"error": "invalid body"})
		return
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if body.Index == nil || *body.Index < 0 || *body.Index >= len(h.cfg.BedrockKey) {
		c.JSON(404, gin.H{"error": "item not found"})
		return
	}
	targetIndex := *body.Index

	entry := h.cfg.BedrockKey[targetIndex]
	if body.Value.APIKey != nil {
		entry.APIKey = strings.TrimSpace(*body.Value.APIKey)
	}
	if body.Value.Profile != nil {
		entry.Profile = strings.TrimSpace(*body.Value.Profile)
	}
	if body.Value.Region != nil {
		entry.Region = strings.TrimSpace(*body.Value.Region)
	}
	if body.Value.Endpoint != nil {
		entry.Endpoint = config.NormalizeBedrockEndpoint(*body.Value.Endpoint)
	}
	if body.Value.BaseURL != nil {
		// An empty override re-derives the URL from endpoint and region on sanitize.
		entry.BaseURL = strings.TrimSpace(*body.Value.BaseURL)
	}
	if body.Value.ChatCompletionsPath != nil {
		entry.ChatCompletionsPath = strings.TrimSpace(*body.Value.ChatCompletionsPath)
	}
	if body.Value.Priority != nil {
		entry.Priority = *body.Value.Priority
	}
	if len(body.Value.Weight) > 0 {
		weight, errWeight := parseCredentialWeightPatch(body.Value.Weight)
		if errWeight != nil {
			c.JSON(400, gin.H{"error": errWeight.Error()})
			return
		}
		entry.Weight = weight
	}
	if body.Value.Prefix != nil {
		entry.Prefix = strings.TrimSpace(*body.Value.Prefix)
	}
	if body.Value.ProxyURL != nil {
		entry.ProxyURL = strings.TrimSpace(*body.Value.ProxyURL)
	}
	if body.Value.Models != nil {
		entry.Models = append([]config.BedrockModel(nil), (*body.Value.Models)...)
	}
	if body.Value.Headers != nil {
		entry.Headers = config.NormalizeHeaders(*body.Value.Headers)
	}
	if body.Value.ExcludedModels != nil {
		entry.ExcludedModels = config.NormalizeExcludedModels(*body.Value.ExcludedModels)
	}
	if !applyDisableCoolingPatch(c, body.Value.DisableCooling, &entry.DisableCooling) {
		return
	}
	if body.Value.RequestRetry != nil {
		entry.RequestRetry = body.Value.RequestRetry
	}
	if body.Value.RequestScopedErrors != nil {
		entry.RequestScopedErrors = append([]config.RequestScopedErrorRule(nil), (*body.Value.RequestScopedErrors)...)
	}
	h.cfg.BedrockKey[targetIndex] = entry
	h.cfg.SanitizeBedrockKeys()
	h.persistLocked(c)
}

// DeleteBedrockKey removes an entry by index, or by profile / api-key match.
func (h *Handler) DeleteBedrockKey(c *gin.Context) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if idxStr := c.Query("index"); idxStr != "" {
		var idx int
		_, errScan := fmt.Sscanf(idxStr, "%d", &idx)
		if errScan == nil && idx >= 0 && idx < len(h.cfg.BedrockKey) {
			h.cfg.BedrockKey = append(h.cfg.BedrockKey[:idx], h.cfg.BedrockKey[idx+1:]...)
			h.cfg.SanitizeBedrockKeys()
			h.persistLocked(c)
			return
		}
	}
	profile := strings.TrimSpace(c.Query("profile"))
	apiKey := strings.TrimSpace(c.Query("api-key"))
	if profile != "" || apiKey != "" {
		matchIndex := -1
		matchCount := 0
		for i := range h.cfg.BedrockKey {
			entry := h.cfg.BedrockKey[i]
			if (profile != "" && strings.TrimSpace(entry.Profile) != profile) || (apiKey != "" && strings.TrimSpace(entry.APIKey) != apiKey) {
				continue
			}
			matchCount++
			if matchIndex == -1 {
				matchIndex = i
			}
		}
		if matchCount > 1 {
			c.JSON(400, gin.H{"error": "multiple items match; use index"})
			return
		}
		if matchIndex != -1 {
			h.cfg.BedrockKey = append(h.cfg.BedrockKey[:matchIndex], h.cfg.BedrockKey[matchIndex+1:]...)
		}
		h.cfg.SanitizeBedrockKeys()
		h.persistLocked(c)
		return
	}
	c.JSON(400, gin.H{"error": "missing index, profile, or api-key"})
}

func normalizeBedrockKey(entry *config.BedrockKey) {
	if entry == nil {
		return
	}
	entry.APIKey = strings.TrimSpace(entry.APIKey)
	entry.Profile = strings.TrimSpace(entry.Profile)
	entry.Region = strings.TrimSpace(entry.Region)
	entry.Endpoint = config.NormalizeBedrockEndpoint(entry.Endpoint)
	entry.BaseURL = strings.TrimSpace(entry.BaseURL)
	entry.ChatCompletionsPath = strings.TrimSpace(entry.ChatCompletionsPath)
	entry.Prefix = strings.TrimSpace(entry.Prefix)
	entry.ProxyURL = strings.TrimSpace(entry.ProxyURL)
	entry.Headers = config.NormalizeHeaders(entry.Headers)
	entry.ExcludedModels = config.NormalizeExcludedModels(entry.ExcludedModels)
	if len(entry.Models) == 0 {
		return
	}
	normalized := make([]config.BedrockModel, 0, len(entry.Models))
	for i := range entry.Models {
		model := entry.Models[i]
		model.Name = strings.TrimSpace(model.Name)
		model.Alias = strings.TrimSpace(model.Alias)
		if model.Name == "" && model.Alias == "" {
			continue
		}
		normalized = append(normalized, model)
	}
	entry.Models = normalized
}
