package helps

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

// awsCredentialsResolveTimeout bounds credential acquisition (SSO, assume-role,
// IMDS). This is a credential-acquisition timeout, which the project allows.
const awsCredentialsResolveTimeout = 30 * time.Second

// AWSCredentialsCache caches one aws.CredentialsProvider per AWS profile so that
// SSO / assume-role / IMDS lookups are performed once and refreshed automatically.
type AWSCredentialsCache struct {
	mu        sync.Mutex
	providers map[string]aws.CredentialsProvider
	loadFn    func(ctx context.Context, profile, region string) (aws.CredentialsProvider, error)
}

// NewAWSCredentialsCache constructs a credentials cache backed by the AWS default
// credential chain (shared config profile, environment, IMDS, SSO, ...).
func NewAWSCredentialsCache() *AWSCredentialsCache {
	return &AWSCredentialsCache{
		providers: make(map[string]aws.CredentialsProvider),
		loadFn:    loadAWSCredentialsProvider,
	}
}

// NewAWSCredentialsCacheWithLoader constructs a cache with a custom loader (tests).
func NewAWSCredentialsCacheWithLoader(loadFn func(ctx context.Context, profile, region string) (aws.CredentialsProvider, error)) *AWSCredentialsCache {
	return &AWSCredentialsCache{providers: make(map[string]aws.CredentialsProvider), loadFn: loadFn}
}

func loadAWSCredentialsProvider(ctx context.Context, profile, region string) (aws.CredentialsProvider, error) {
	opts := make([]func(*awsconfig.LoadOptions) error, 0, 2)
	if profile != "" {
		opts = append(opts, awsconfig.WithSharedConfigProfile(profile))
	}
	if region != "" {
		opts = append(opts, awsconfig.WithRegion(region))
	}
	cfg, err := awsconfig.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("load aws config (profile %q): %w", profile, err)
	}
	if cfg.Credentials == nil {
		return nil, fmt.Errorf("aws profile %q resolved no credentials provider", profile)
	}
	return aws.NewCredentialsCache(cfg.Credentials), nil
}

// Retrieve returns AWS credentials for the given profile, loading and caching the
// provider on first use. An empty profile selects the default credential chain.
func (c *AWSCredentialsCache) Retrieve(ctx context.Context, profile, region string) (aws.Credentials, error) {
	if c == nil {
		return aws.Credentials{}, fmt.Errorf("aws credentials cache is nil")
	}
	profile = strings.TrimSpace(profile)
	region = strings.TrimSpace(region)
	cacheKey := profile + "|" + region

	c.mu.Lock()
	provider, ok := c.providers[cacheKey]
	c.mu.Unlock()
	if !ok {
		loadCtx, cancel := context.WithTimeout(ctx, awsCredentialsResolveTimeout)
		loaded, err := c.loadFn(loadCtx, profile, region)
		cancel()
		if err != nil {
			return aws.Credentials{}, err
		}
		c.mu.Lock()
		if existing, exists := c.providers[cacheKey]; exists {
			loaded = existing
		} else {
			c.providers[cacheKey] = loaded
		}
		c.mu.Unlock()
		provider = loaded
	}

	retrieveCtx, cancel := context.WithTimeout(ctx, awsCredentialsResolveTimeout)
	defer cancel()
	creds, err := provider.Retrieve(retrieveCtx)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("retrieve aws credentials (profile %q): %w", profile, err)
	}
	return creds, nil
}

// Invalidate drops the cached provider for a profile so the next Retrieve reloads it.
func (c *AWSCredentialsCache) Invalidate(profile, region string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	delete(c.providers, strings.TrimSpace(profile)+"|"+strings.TrimSpace(region))
	c.mu.Unlock()
}

// SignAWSRequest signs req in place with AWS SigV4 for the given service and
// region. The request body is read (and restored) to compute the payload hash,
// so callers must set req.Body before signing and must not mutate headers afterwards.
func SignAWSRequest(ctx context.Context, req *http.Request, creds aws.Credentials, service, region string) error {
	if req == nil {
		return fmt.Errorf("sign aws request: request is nil")
	}
	payloadHash, err := awsPayloadHash(req)
	if err != nil {
		return err
	}
	// Go's transport adds Accept-Encoding automatically; strip it so it isn't
	// added after signing without being part of the signature.
	req.Header.Del("Accept-Encoding")
	signer := v4.NewSigner()
	if errSign := signer.SignHTTP(ctx, creds, req, payloadHash, service, region, time.Now().UTC()); errSign != nil {
		return fmt.Errorf("sign aws request: %w", errSign)
	}
	return nil
}

func awsPayloadHash(req *http.Request) (string, error) {
	if req.Body == nil || req.Body == http.NoBody {
		sum := sha256.Sum256(nil)
		return hex.EncodeToString(sum[:]), nil
	}
	body, err := io.ReadAll(req.Body)
	if errClose := req.Body.Close(); errClose != nil {
		return "", fmt.Errorf("sign aws request: close body: %w", errClose)
	}
	if err != nil {
		return "", fmt.Errorf("sign aws request: read body: %w", err)
	}
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:]), nil
}
