package main

import (
	"bufio"
	"crypto/subtle"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	listenAddr   = flag.String("listen", ":8501", "listen address")
	streamlitURL = flag.String("streamlit", "http://127.0.0.1:8503", "streamlit upstream")
	htmlPath     = flag.String("html", "igv_viewer.html", "path to IGV HTML file")
	mappingPath  = flag.String("mapping", "mapping.tsv", "path to TSV mapping file")
	tlsCertPath  = flag.String("tls-cert", "/move/certs/server.crt", "path to TLS certificate (falls back to plain HTTP if missing)")
	tlsKeyPath   = flag.String("tls-key", "/move/certs/server.key", "path to TLS private key (falls back to plain HTTP if missing)")
	debug        = flag.Bool("debug", false, "enable debug-level logging (request headers, auth/lockout detail, byte-range parsing)")
)

// Set by loadAuthCredentials from the MOVE_USERNAME / MOVE_PASSWORD
// environment variables at startup.
var (
	validUsername string
	validPassword string
)

func loadAuthCredentials() {
	validUsername = os.Getenv("MOVE_USERNAME")
	validPassword = os.Getenv("MOVE_PASSWORD")
	if validUsername == "" || validPassword == "" {
		slog.Error("MOVE_USERNAME and MOVE_PASSWORD environment variables must both be set")
		os.Exit(1)
	}
}

// Basic Auth brute-force protection: an IP is locked out for
// lockoutDuration after maxFailedAttempts failures within lockoutWindow.
const (
	maxFailedAttempts = 5
	lockoutWindow     = 5 * time.Minute
	lockoutDuration   = 15 * time.Minute
	cleanupInterval   = 10 * time.Minute
)

type attemptRecord struct {
	count       int
	lastAttempt time.Time
	lockedUntil time.Time
}

type loginAttempts struct {
	mu       sync.Mutex
	failures map[string]*attemptRecord
}

func newLoginAttempts() *loginAttempts {
	la := &loginAttempts{failures: make(map[string]*attemptRecord)}
	go la.cleanupLoop()
	return la
}

func (la *loginAttempts) lockedFor(key string) time.Duration {
	la.mu.Lock()
	defer la.mu.Unlock()

	rec, ok := la.failures[key]
	if !ok {
		return 0
	}
	if remaining := time.Until(rec.lockedUntil); remaining > 0 {
		return remaining
	}
	return 0
}

func (la *loginAttempts) recordFailure(key string) {
	la.mu.Lock()
	defer la.mu.Unlock()

	now := time.Now()
	rec, ok := la.failures[key]
	if !ok || now.Sub(rec.lastAttempt) > lockoutWindow {
		rec = &attemptRecord{}
		la.failures[key] = rec
	}

	rec.count++
	rec.lastAttempt = now
	if rec.count >= maxFailedAttempts {
		rec.lockedUntil = now.Add(lockoutDuration)
	}
}

func (la *loginAttempts) recordSuccess(key string) {
	la.mu.Lock()
	defer la.mu.Unlock()
	delete(la.failures, key)
}

func (la *loginAttempts) cleanupLoop() {
	for {
		time.Sleep(cleanupInterval)

		la.mu.Lock()
		now := time.Now()
		for key, rec := range la.failures {
			if now.After(rec.lockedUntil) && now.Sub(rec.lastAttempt) > lockoutWindow {
				delete(la.failures, key)
			}
		}
		la.mu.Unlock()
	}
}

// clientKey identifies the caller for lockout purposes. It uses the TCP
// peer address rather than a client-supplied header (e.g. X-Forwarded-For)
// since this proxy is the first hop and such headers can be spoofed.
func clientKey(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

type Server struct {
	sampleToCRAM map[string]string
}

func setupLogging() {
	level := slog.LevelInfo
	if *debug {
		level = slog.LevelDebug
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})
	slog.SetDefault(slog.New(handler))
}

func main() {
	flag.Parse()
	setupLogging()
	loadAuthCredentials()

	streamlit, err := url.Parse(*streamlitURL)
	if err != nil {
		slog.Error("invalid streamlit URL", "url", *streamlitURL, "error", err)
		os.Exit(1)
	}

	srvState, err := loadMapping(*mappingPath)
	if err != nil {
		slog.Error("failed to load mapping", "path", *mappingPath, "error", err)
		os.Exit(1)
	}

	streamlitProxy := newProxy(streamlit)

	mux := http.NewServeMux()

	mux.HandleFunc("/igvjs/", func(w http.ResponseWriter, r *http.Request) {
		slog.Info("IGVHTML", "method", r.Method, "url", r.URL.String())
		slog.Debug("request headers", "headers", redactedHeaders(r.Header))

		data, err := os.ReadFile(*htmlPath)
		if err != nil {
			slog.Error("failed to read HTML file", "path", *htmlPath, "error", err)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(data)
	})

	mux.HandleFunc("/alignments/", func(w http.ResponseWriter, r *http.Request) {
		slog.Info("CRAM/CRAI", "method", r.Method, "url", r.URL.String())
		slog.Debug("request headers", "headers", redactedHeaders(r.Header))
		serveAlignmentFile(w, r, srvState.sampleToCRAM)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		slog.Info("STREAM", "method", r.Method, "url", r.URL.String())
		slog.Debug("request headers", "headers", redactedHeaders(r.Header))
		streamlitProxy.ServeHTTP(w, r)
	})

	protectedMux := basicAuth(mux)

	server := &http.Server{
		Addr:              *listenAddr,
		Handler:           protectedMux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	slog.Info("proxy listening", "addr", *listenAddr)
	slog.Info("streamlit upstream", "url", *streamlitURL)
	slog.Info("html file", "path", *htmlPath)
	slog.Info("mapping file", "path", *mappingPath)
	slog.Info("debug logging", "enabled", *debug)

	if certFileExists(*tlsCertPath) && certFileExists(*tlsKeyPath) {
		slog.Info("TLS enabled", "cert", *tlsCertPath)
		slog.Error("server exited", "error", server.ListenAndServeTLS(*tlsCertPath, *tlsKeyPath))
		os.Exit(1)
	}

	slog.Info("TLS cert/key not found, falling back to plain HTTP", "cert", *tlsCertPath, "key", *tlsKeyPath)
	slog.Error("server exited", "error", server.ListenAndServe())
	os.Exit(1)
}

func certFileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func basicAuth(next http.Handler) http.Handler {
	attempts := newLoginAttempts()

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := clientKey(r)

		if remaining := attempts.lockedFor(key); remaining > 0 {
			slog.Debug("request rejected: client locked out", "client", key, "retry_after", remaining)
			w.Header().Set("Retry-After", strconv.Itoa(int(remaining.Seconds())))
			http.Error(w, "too many failed login attempts, try again later", http.StatusTooManyRequests)
			return
		}

		// Deliberately not logging username/password here, even at debug
		// level -- Basic Auth credentials shouldn't end up in log output.
		username, password, ok := r.BasicAuth()
		if !ok || !validCredentials(username, password) {
			attempts.recordFailure(key)
			slog.Debug("auth failed", "client", key, "credentials_supplied", ok)
			w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		attempts.recordSuccess(key)
		slog.Debug("auth succeeded", "client", key)
		next.ServeHTTP(w, r)
	})
}

func validCredentials(username, password string) bool {
	userOK := subtle.ConstantTimeCompare([]byte(username), []byte(validUsername)) == 1
	passOK := subtle.ConstantTimeCompare([]byte(password), []byte(validPassword)) == 1
	return userOK && passOK
}

func loadMapping(tsvPath string) (*Server, error) {
	f, err := os.Open(tsvPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	lines := bufio.NewScanner(f)
	lines.Buffer(make([]byte, 1024), 1024*1024)

	var header []string
	mapping := make(map[string]string)

	first := true
	for lines.Scan() {
		line := strings.TrimSpace(lines.Text())
		if line == "" {
			continue
		}

		fields := strings.Split(line, "\t")

		if first {
			header = fields
			first = false
			continue
		}

		if len(fields) < len(header) {
			continue
		}

		var sample, cram string
		for i, h := range header {
			switch strings.TrimSpace(h) {
			case "sample_name":
				sample = strings.TrimSpace(fields[i])
			case "cram_path":
				cram = strings.TrimSpace(fields[i])
			}
		}

		if sample != "" && cram != "" {
			mapping[sample] = cram
			slog.Debug("mapping entry loaded", "sample", sample, "cram", cram)
		}
	}

	if err := lines.Err(); err != nil {
		return nil, err
	}

	slog.Info("mapping loaded", "path", tsvPath, "samples", len(mapping))
	return &Server{sampleToCRAM: mapping}, nil
}

func serveAlignmentFile(w http.ResponseWriter, r *http.Request, mapping map[string]string) {
	addCORSHeaders(w)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/alignments/")
	path = strings.Trim(path, "/")

	if path == "" {
		http.Error(w, "missing sample name", http.StatusBadRequest)
		return
	}

	isIndex := false
	sample := path

	if strings.HasSuffix(path, ".crai") {
		isIndex = true
		sample = strings.TrimSuffix(path, ".crai")
	}

	cramPath, ok := mapping[sample]
	if !ok {
		slog.Debug("unknown sample requested", "sample", sample)
		http.Error(w, "unknown sample: "+sample, http.StatusNotFound)
		return
	}

	filePath := cramPath
	if isIndex {
		filePath = craiPathForCRAM(cramPath)
	}

	info, err := os.Stat(filePath)
	if err != nil {
		slog.Debug("file not found", "sample", sample, "path", filePath, "error", err)
		http.Error(w, "file not found for sample: "+sample, http.StatusNotFound)
		return
	}
	if info.IsDir() {
		http.Error(w, "not a file", http.StatusNotFound)
		return
	}

	fileSize := info.Size()
	mimeType := mime.TypeByExtension(filepath.Ext(filePath))
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	slog.Debug("serving alignment file", "sample", sample, "path", filePath, "size", fileSize, "is_index", isIndex)

	rangeHeader := r.Header.Get("Range")
	if rangeHeader != "" {
		start, end, ok := parseByteRange(rangeHeader, fileSize)
		if !ok {
			slog.Debug("invalid range header", "sample", sample, "range", rangeHeader)
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", fileSize))
			http.Error(w, "invalid range", http.StatusRequestedRangeNotSatisfiable)
			return
		}
		slog.Debug("serving byte range", "sample", sample, "start", start, "end", end)

		w.Header().Set("Content-Type", mimeType)
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, fileSize))
		w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
		w.WriteHeader(http.StatusPartialContent)

		if r.Method == http.MethodHead {
			return
		}

		f, err := os.Open(filePath)
		if err != nil {
			http.Error(w, "file open error", http.StatusInternalServerError)
			return
		}
		defer f.Close()

		_, _ = f.Seek(start, io.SeekStart)
		_, _ = io.CopyN(w, f, end-start+1)
		return
	}

	w.Header().Set("Content-Type", mimeType)
	w.Header().Set("Content-Length", strconv.FormatInt(fileSize, 10))
	w.WriteHeader(http.StatusOK)

	if r.Method == http.MethodHead {
		return
	}

	f, err := os.Open(filePath)
	if err != nil {
		http.Error(w, "file open error", http.StatusInternalServerError)
		return
	}
	defer f.Close()

	_, _ = io.Copy(w, f)
}

func craiPathForCRAM(cramPath string) string {
	// Prefer the file.cram.crai form (index alongside the CRAM with its
	// extension kept intact), since that's what most alignment pipelines
	// produce. Fall back to the file.crai form (extension replaced) for
	// older layouts that only have that file.
	withSuffix := cramPath + ".crai"
	if _, err := os.Stat(withSuffix); err == nil {
		return withSuffix
	}

	ext := filepath.Ext(cramPath)
	if ext == "" {
		return withSuffix
	}
	return strings.TrimSuffix(cramPath, ext) + ".crai"
}

func parseByteRange(header string, size int64) (start, end int64, ok bool) {
	if !strings.HasPrefix(header, "bytes=") {
		return 0, 0, false
	}

	spec := strings.TrimPrefix(header, "bytes=")
	parts := strings.SplitN(spec, "-", 2)
	if len(parts) != 2 {
		return 0, 0, false
	}

	startStr := strings.TrimSpace(parts[0])
	endStr := strings.TrimSpace(parts[1])

	if startStr == "" {
		suffixLen, err := strconv.ParseInt(endStr, 10, 64)
		if err != nil || suffixLen <= 0 {
			return 0, 0, false
		}
		if suffixLen > size {
			suffixLen = size
		}
		return size - suffixLen, size - 1, true
	}

	start, err := strconv.ParseInt(startStr, 10, 64)
	if err != nil || start < 0 || start >= size {
		return 0, 0, false
	}

	if endStr == "" {
		return start, size - 1, true
	}

	end, err = strconv.ParseInt(endStr, 10, 64)
	if err != nil || end < start {
		return 0, 0, false
	}
	if end >= size {
		end = size - 1
	}

	return start, end, true
}

func addCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Range, Content-Type, Authorization")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
	w.Header().Set("Accept-Ranges", "bytes")
}

func newProxy(target *url.URL) *httputil.ReverseProxy {
	// Note: intentionally not rewriting r.Host to the streamlit upstream
	// here. Streamlit's WebSocket handler falls back to Tornado's default
	// check_origin, which compares the Origin header against the request's
	// Host header -- if Host were rewritten to the upstream address, it
	// would never match the browser's Origin (the proxy's public address),
	// and Streamlit rejects the /_stcore/stream WebSocket connection.
	proxy := httputil.NewSingleHostReverseProxy(target)

	proxy.ModifyResponse = func(resp *http.Response) error {
		slog.Debug("proxy response", "status", resp.StatusCode, "url", resp.Request.URL.String())
		addCORSHeadersToResponse(resp)
		return nil
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		slog.Error("proxy error", "method", r.Method, "url", r.URL.String(), "error", err)
		addCORSHeaders(w)
		http.Error(w, "bad gateway", http.StatusBadGateway)
	}

	return proxy
}

// redactedHeaders returns a copy of h with sensitive header values masked,
// safe to pass to a debug log even though it may contain the raw
// Authorization header used for Basic Auth.
func redactedHeaders(h http.Header) http.Header {
	redacted := h.Clone()
	if redacted.Get("Authorization") != "" {
		redacted.Set("Authorization", "REDACTED")
	}
	return redacted
}

func addCORSHeadersToResponse(resp *http.Response) {
	resp.Header.Set("Access-Control-Allow-Origin", "*")
	resp.Header.Set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
	resp.Header.Set("Access-Control-Allow-Headers", "Range, Content-Type, Authorization")
	resp.Header.Set("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges")
	resp.Header.Set("Accept-Ranges", "bytes")
}
