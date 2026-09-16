package managementasset

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

func TestFetchLatestAssetSetsGitHubAuthorization(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "asset-token")
	t.Setenv("GITSTORE_GIT_TOKEN", "")
	t.Setenv("GITSTORE_GIT_URL", "")

	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		authorization = req.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"assets":[{"name":"management.html","browser_download_url":"https://example.com/management.html","digest":"sha256:abc123"}]}`))
	}))
	defer server.Close()

	asset, remoteHash, err := fetchLatestAsset(t.Context(), server.Client(), server.URL)
	if err != nil {
		t.Fatalf("fetchLatestAsset() error = %v", err)
	}
	if authorization != "Bearer asset-token" {
		t.Fatalf("Authorization = %q, want %q", authorization, "Bearer asset-token")
	}
	if asset == nil || asset.Name != managementAssetName {
		t.Fatalf("asset = %#v, want %q", asset, managementAssetName)
	}
	if remoteHash != "abc123" {
		t.Fatalf("remoteHash = %q, want %q", remoteHash, "abc123")
	}
}

func TestFetchLatestAssetOmitsAuthorizationWithoutToken(t *testing.T) {
	t.Setenv("GITHUB_TOKEN", "")
	t.Setenv("github_token", "")
	t.Setenv("GITSTORE_GIT_TOKEN", "")
	t.Setenv("GITSTORE_GIT_URL", "")

	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		authorization = req.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"assets":[{"name":"management.html","browser_download_url":"https://example.com/management.html","digest":"sha256:abc123"}]}`))
	}))
	defer server.Close()

	asset, remoteHash, err := fetchLatestAsset(t.Context(), server.Client(), server.URL)
	if err != nil {
		t.Fatalf("fetchLatestAsset() error = %v", err)
	}
	if authorization != "" {
		t.Fatalf("Authorization = %q, want empty", authorization)
	}
	if asset == nil || asset.Name != managementAssetName {
		t.Fatalf("asset = %#v, want %q", asset, managementAssetName)
	}
	if remoteHash != "abc123" {
		t.Fatalf("remoteHash = %q, want %q", remoteHash, "abc123")
	}
}

func TestAutoUpdateSkipReason(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *config.Config
		wantReason string
		wantSkip   bool
	}{
		{
			name:       "nil config",
			cfg:        nil,
			wantReason: "config not yet available",
			wantSkip:   true,
		},
		{
			name: "cluster mode",
			cfg: &config.Config{
				Home: config.HomeConfig{Enabled: true},
			},
			wantReason: "cluster mode enabled",
			wantSkip:   true,
		},
		{
			name: "control panel disabled",
			cfg: &config.Config{
				RemoteManagement: config.RemoteManagement{DisableControlPanel: true},
			},
			wantReason: "control panel disabled",
			wantSkip:   true,
		},
		{
			name: "local panel path set",
			cfg: &config.Config{
				RemoteManagement: config.RemoteManagement{PanelLocalPath: "/tmp/panel"},
			},
			wantReason: "panel-local-path is set",
			wantSkip:   true,
		},
		{
			name: "auto update disabled",
			cfg: &config.Config{
				RemoteManagement: config.RemoteManagement{DisableAutoUpdatePanel: true},
			},
			wantReason: "disable-auto-update-panel is enabled",
			wantSkip:   true,
		},
		{
			name:       "enabled",
			cfg:        &config.Config{},
			wantReason: "",
			wantSkip:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotReason, gotSkip := autoUpdateSkipReason(tt.cfg)
			if gotReason != tt.wantReason || gotSkip != tt.wantSkip {
				t.Fatalf("autoUpdateSkipReason() = (%q, %t), want (%q, %t)", gotReason, gotSkip, tt.wantReason, tt.wantSkip)
			}
		})
	}
}

func TestLocalPanelFile(t *testing.T) {
	if got := LocalPanelFile("  "); got != "" {
		t.Fatalf("empty path resolved to %q", got)
	}
	if got := LocalPanelFile(filepath.Join(t.TempDir(), "missing")); got != "" {
		t.Fatalf("missing path resolved to %q", got)
	}

	repoDir := t.TempDir()
	if got := LocalPanelFile(repoDir); got != "" {
		t.Fatalf("directory without a panel resolved to %q", got)
	}
	distFile := filepath.Join(repoDir, "dist", "index.html")
	if err := os.MkdirAll(filepath.Dir(distFile), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(distFile, []byte("<html></html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := LocalPanelFile(repoDir); got != distFile {
		t.Fatalf("checked-out repo resolved to %q, want %q", got, distFile)
	}

	// Vite keeps a source index.html at the repo root; the built file must still win.
	sourceIndex := filepath.Join(repoDir, "index.html")
	if err := os.WriteFile(sourceIndex, []byte("<script src=\"/src/main.tsx\"></script>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := LocalPanelFile(repoDir); got != distFile {
		t.Fatalf("repo with source index.html resolved to %q, want %q", got, distFile)
	}

	// A management.html directly in the directory wins over dist/index.html.
	direct := filepath.Join(repoDir, ManagementFileName)
	if err := os.WriteFile(direct, []byte("<html></html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := LocalPanelFile(repoDir); got != direct {
		t.Fatalf("directory resolved to %q, want %q", got, direct)
	}
	if got := LocalPanelFile(distFile); got != distFile {
		t.Fatalf("explicit file resolved to %q", got)
	}
}
