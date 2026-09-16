package helps

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
)

func TestSignAWSRequestAddsSigV4Headers(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/anthropic/v1/messages", strings.NewReader(`{"hello":"world"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	creds := aws.Credentials{AccessKeyID: "AKIAEXAMPLE", SecretAccessKey: "secret", SessionToken: "session"}
	if errSign := SignAWSRequest(context.Background(), req, creds, "bedrock", "us-east-1"); errSign != nil {
		t.Fatalf("SignAWSRequest error = %v", errSign)
	}
	authz := req.Header.Get("Authorization")
	if !strings.HasPrefix(authz, "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/") {
		t.Fatalf("unexpected Authorization header: %q", authz)
	}
	if !strings.Contains(authz, "/us-east-1/bedrock/aws4_request") {
		t.Fatalf("credential scope missing service/region: %q", authz)
	}
	if req.Header.Get("X-Amz-Date") == "" {
		t.Fatal("X-Amz-Date header missing")
	}
	if req.Header.Get("X-Amz-Security-Token") != "session" {
		t.Fatalf("session token header = %q", req.Header.Get("X-Amz-Security-Token"))
	}
	// Body must still be readable after signing.
	buf := make([]byte, 64)
	n, _ := req.Body.Read(buf)
	if string(buf[:n]) != `{"hello":"world"}` {
		t.Fatalf("body not restored after signing: %q", string(buf[:n]))
	}
}

func TestAWSCredentialsCacheLoadsOncePerProfile(t *testing.T) {
	loads := 0
	cache := NewAWSCredentialsCacheWithLoader(func(ctx context.Context, profile, region string) (aws.CredentialsProvider, error) {
		loads++
		if profile == "broken" {
			return nil, errors.New("boom")
		}
		return credentials.NewStaticCredentialsProvider("AKIA"+profile, "secret", ""), nil
	})
	for i := 0; i < 3; i++ {
		creds, err := cache.Retrieve(context.Background(), "dev", "us-east-1")
		if err != nil {
			t.Fatalf("Retrieve error = %v", err)
		}
		if creds.AccessKeyID != "AKIAdev" {
			t.Fatalf("access key = %q", creds.AccessKeyID)
		}
	}
	if loads != 1 {
		t.Fatalf("loader called %d times, want 1", loads)
	}
	if _, err := cache.Retrieve(context.Background(), "broken", ""); err == nil {
		t.Fatal("expected error for broken profile")
	}
	cache.Invalidate("dev", "us-east-1")
	if _, err := cache.Retrieve(context.Background(), "dev", "us-east-1"); err != nil {
		t.Fatalf("Retrieve after invalidate error = %v", err)
	}
	if loads != 3 {
		t.Fatalf("loader called %d times, want 3", loads)
	}
}
