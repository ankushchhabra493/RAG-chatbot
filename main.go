package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "math"
    "net/http"
    "strings"
    "time"
    "bytes"
    "io/ioutil"

    "github.com/google/generative-ai-go/genai"
    "google.golang.org/api/googleapi"
    "google.golang.org/api/option"
)

const (
    GEMINI_API_KEY = "AIzaSyB8VhcP_Sfghc_TbwQVZm_R5x1_IxVYhM8"
    CHROMA_API_URL = "http://localhost:8000/api/v1/query" // Use v1 endpoint for compatibility
    CHROMA_HEARTBEAT_URL = "http://localhost:8000/api/v1/heartbeat"
    CHROMA_COLLECTION = "pubmed-central"
)

type ChatRequest struct {
    Message string `json:"message"`
}

type ChatResponse struct {
    Response string `json:"response"`
}

var client *genai.Client
var model *genai.GenerativeModel

func init() {
    ctx := context.Background()
    var err error
    client, err = genai.NewClient(ctx, option.WithAPIKey(GEMINI_API_KEY))
    if err != nil {
        log.Fatalf("Failed to create client: %v", err)
    }
    
    // FIX: assign to the global model variable, not a new local one
    model = client.GenerativeModel("models/gemini-1.5-flash")
    log.Printf("Initialized model: gemini-1.5-flash")
    
    // Configure model settings if needed
    model.SetTemperature(0.7)
}

func sendMessageWithRetry(ctx context.Context, cs *genai.ChatSession, message string) (*genai.GenerateContentResponse, error) {
    maxRetries := 3
    var resp *genai.GenerateContentResponse
    var err error
    
    for attempt := 0; attempt < maxRetries; attempt++ {
        if attempt > 0 {
            // Exponential backoff: 1s, 2s, 4s...
            backoffTime := time.Duration(math.Pow(2, float64(attempt-1))) * time.Second
            log.Printf("Rate limited. Retrying in %v (attempt %d/%d)", backoffTime, attempt+1, maxRetries)
            time.Sleep(backoffTime)
        }
        
        resp, err = cs.SendMessage(ctx, genai.Text(message))
        if err == nil {
            return resp, nil
        }
        
        // Check if it's a rate limit error
        if strings.Contains(err.Error(), "429") {
            log.Printf("Rate limited on attempt %d: %v", attempt+1, err)
            // Continue to retry
            continue
        }
        
        // If it's not a rate limit error, don't retry
        return nil, err
    }
    
    return nil, fmt.Errorf("exceeded maximum retries (%d): %w", maxRetries, err)
}

func enableCORS(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }

        next(w, r)
    }
}

// Helper: Get embedding for a query using Gemini
func getEmbedding(ctx context.Context, text string) ([]float32, error) {
    log.Printf("Step 2.1: Preparing embedding request to Python server")
    payload := fmt.Sprintf(`{"text": %q}`, text)
    req, err := http.NewRequest("POST", "http://localhost:9000/embed", bytes.NewBuffer([]byte(payload)))
    if err != nil {
        log.Printf("Step 2.2: Error creating embedding request: %v", err)
        return nil, err
    }
    req.Header.Set("Content-Type", "application/json")
    client := &http.Client{}
    log.Printf("Step 2.3: Sending embedding request to Python server")
    resp, err := client.Do(req)
    if err != nil {
        log.Printf("Step 2.4: Error sending embedding request: %v", err)
        log.Printf("HINT: Make sure your Python embedding server is running with: uvicorn embed_server:app --host 0.0.0.0 --port 9000")
        return nil, err
    }
    defer resp.Body.Close()
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        log.Printf("Step 2.5: Error reading embedding response: %v", err)
        return nil, err
    }
    var pyResp struct {
        Embedding []float32 `json:"embedding"`
    }
    if err := json.Unmarshal(body, &pyResp); err != nil {
        log.Printf("Step 2.6: Error unmarshalling embedding response: %v", err)
        return nil, err
    }
    log.Printf("Step 2.7: Successfully received embedding from Python server")
    return pyResp.Embedding, nil
}

// Helper: Query ChromaDB for relevant contexts
func queryChromaDB(queryEmbedding []float32, topK int) ([]string, error) {
    log.Printf("Step 3.1: Preparing ChromaDB query payload")
    // Prepare request payload for v1 API
    payload := fmt.Sprintf(`{
        "collection": "%s",
        "query_embeddings": [%v],
        "n_results": %d
    }`, CHROMA_COLLECTION, floatArrayToString(queryEmbedding), topK)

    req, err := http.NewRequest("POST", CHROMA_API_URL, bytes.NewBuffer([]byte(payload)))
    if err != nil {
        log.Printf("Step 3.2: Error creating ChromaDB request: %v", err)
        return nil, err
    }
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    log.Printf("Step 3.3: Sending request to ChromaDB (v1 API)")
    resp, err := client.Do(req)
    if err != nil {
        log.Printf("Step 3.4: Error sending request to ChromaDB: %v", err)
        log.Printf("HINT: If you are not using Docker, start ChromaDB with: chroma run")
        log.Printf("      Or see https://docs.trychroma.com/getting-started for setup instructions.")
        return nil, err
    }
    defer resp.Body.Close()

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        log.Printf("Step 3.5: Error reading ChromaDB response: %v", err)
        return nil, err
    }

    // Log HTTP status and body for debugging
    log.Printf("Step 3.5b: ChromaDB HTTP status: %d", resp.StatusCode)
    log.Printf("Step 3.5c: ChromaDB raw response: %s", string(body))

    if resp.StatusCode == 405 {
        log.Printf("HINT: ChromaDB returned 405 (Method Not Allowed). This usually means the endpoint does not accept POST requests.")
        log.Printf("      Double-check the ChromaDB REST API docs for the correct method and endpoint: https://docs.trychroma.com/reference/rest")
        log.Printf("      Also ensure your ChromaDB server is up-to-date and supports the /api/v1/query POST endpoint.")
        return nil, fmt.Errorf("ChromaDB returned 405 Method Not Allowed for /api/v1/query")
    }
    if resp.StatusCode == 404 {
        log.Printf("HINT: ChromaDB returned 404. This usually means the /api/v1/query endpoint does not exist or is not enabled.")
        log.Printf("      Double-check your ChromaDB version and API docs: https://docs.trychroma.com/reference/rest")
        log.Printf("      Also ensure your ChromaDB server is up-to-date and supports the v1 API.")
        return nil, fmt.Errorf("ChromaDB returned 404 Not Found for /api/v1/query")
    }
    if resp.StatusCode != 200 {
        return nil, fmt.Errorf("ChromaDB returned non-200 status: %d, body: %s", resp.StatusCode, string(body))
    }

    // Parse response (adjust according to your ChromaDB v1 API response)
    var chromaResp struct {
        Results [][]struct {
            Document string `json:"document"`
        } `json:"results"`
    }
    if err := json.Unmarshal(body, &chromaResp); err != nil {
        log.Printf("Step 3.6: Error unmarshalling ChromaDB response: %v", err)
        log.Printf("ChromaDB raw response: %s", string(body))
        return nil, err
    }

    var contexts []string
    if len(chromaResp.Results) > 0 {
        for _, doc := range chromaResp.Results[0] {
            contexts = append(contexts, doc.Document)
        }
        log.Printf("Step 3.7: ChromaDB returned %d contexts", len(contexts))
    } else {
        log.Printf("Step 3.7: ChromaDB returned 0 contexts")
    }
    return contexts, nil
}

// Helper: Convert []float32 to comma-separated string
func floatArrayToString(arr []float32) string {
    var strArr []string
    for _, f := range arr {
        strArr = append(strArr, fmt.Sprintf("%f", f))
    }
    return strings.Join(strArr, ",")
}

func handleChat(w http.ResponseWriter, r *http.Request) {
    log.Printf("---- handleChat called ----")
    // Enable CORS headers for the actual response
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

    if r.Method != http.MethodPost {
        log.Printf("Invalid method: %s", r.Method)
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    log.Printf("Step 1: Decoding user request")
    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        log.Printf("Error decoding request: %v", err)
        http.Error(w, "Invalid request format", http.StatusBadRequest)
        return
    }

    log.Printf("Received message: %s", req.Message)

    ctx := context.Background()

    // === RAG: Retrieve relevant PubMed Central context ===
    log.Printf("Step 2: Getting embedding for user query")
    embedding, err := getEmbedding(ctx, req.Message)
    var retrievedContexts []string
    if err == nil && embedding != nil {
        log.Printf("Step 3: Querying ChromaDB for relevant contexts")
        retrievedContexts, err = queryChromaDB(embedding, 3)
        if err != nil {
            log.Printf("ChromaDB retrieval error: %v", err)
        } else {
            log.Printf("ChromaDB returned %d contexts", len(retrievedContexts))
        }
    } else {
        log.Printf("Embedding error: %v", err)
    }

    // Compose prompt with retrieved context
    log.Printf("Step 4: Building prompt for Gemini")
    var promptBuilder strings.Builder
    if len(retrievedContexts) > 0 {
        promptBuilder.WriteString("You are a medical assistant. Use the following PubMed Central context to answer the question:\n\n")
        for i, ctx := range retrievedContexts {
            promptBuilder.WriteString(fmt.Sprintf("Context %d: %s\n", i+1, ctx))
        }
        promptBuilder.WriteString("\nQuestion: ")
        promptBuilder.WriteString(req.Message)
    } else {
        promptBuilder.WriteString("You are a medical assistant. Answer the following question:\n")
        promptBuilder.WriteString(req.Message)
    }
    finalPrompt := promptBuilder.String()
    log.Printf("Final prompt sent to Gemini:\n%s", finalPrompt)

    // === Gemini LLM Call ===
    log.Printf("Step 5: Sending prompt to Gemini LLM")
    cs := model.StartChat()
    resp, err := sendMessageWithRetry(ctx, cs, finalPrompt)
    
    if err != nil {
        log.Printf("Gemini API error (FULL): %v", err)
        if apiErr, ok := err.(*googleapi.Error); ok {
            log.Printf("Google API Error Code: %d, Message: %s", apiErr.Code, apiErr.Message)
        }
        if strings.Contains(err.Error(), "429") {
            http.Error(w, "Service is currently busy. Please try again in a few moments.", http.StatusTooManyRequests)
        } else if strings.Contains(err.Error(), "404") {
            http.Error(w, "AI model not found. Please check configuration.", http.StatusInternalServerError)
        } else {
            http.Error(w, "Error generating response", http.StatusInternalServerError)
        }
        return
    }

    log.Printf("Step 6: Received response from Gemini")

    if resp == nil || len(resp.Candidates) == 0 {
        log.Printf("Empty response from Gemini API")
        http.Error(w, "Empty response from AI", http.StatusInternalServerError)
        return
    }

    // Improved: Loop through all parts and concatenate text parts
    var response string
    parts := resp.Candidates[0].Content.Parts
    for i, part := range parts {
        switch v := part.(type) {
        case genai.Text:
            log.Printf("Part %d is genai.Text: %s", i, string(v))
            response += string(v)
        default:
            log.Printf("Part %d is of unexpected type: %T", i, v)
        }
    }

    if response == "" {
        log.Printf("Failed to extract text from response")
        http.Error(w, "Empty response from AI", http.StatusInternalServerError)
        return
    }

    log.Printf("Step 7: Sending response to client")
    w.Header().Set("Content-Type", "application/json")
    if err := json.NewEncoder(w).Encode(ChatResponse{Response: response}); err != nil {
        log.Printf("Error encoding response: %v", err)
        http.Error(w, "Error encoding response", http.StatusInternalServerError)
        return
    }
    log.Printf("Successfully sent response to client")
}

// (Optional) Add a health check for ChromaDB v1
func checkChromaDBHealth() {
    resp, err := http.Get(CHROMA_HEARTBEAT_URL)
    if err != nil {
        log.Printf("ChromaDB v1 heartbeat check failed: %v", err)
        return
    }
    defer resp.Body.Close()
    body, _ := ioutil.ReadAll(resp.Body)
    log.Printf("ChromaDB v1 heartbeat response: %s", string(body))
}

func main() {
    // Check ChromaDB v1 health at startup
    checkChromaDBHealth()
    // Handle chat endpoint with CORS
    http.HandleFunc("/chat", enableCORS(handleChat))

    // Serve static files from the "static" directory
    fs := http.FileServer(http.Dir("static"))
    http.Handle("/", fs)

    port := "8080"
    log.Printf("Server starting on http://localhost:%s", port)
    if err := http.ListenAndServe(":"+port, nil); err != nil {
        log.Fatal(err)
    }
}