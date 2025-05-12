package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
)

const (
	GEMINI_API_KEY       = "AIzaSyB8VhcP_Sfghc_TbwQVZm_R5x1_IxVYhM8"
	EMBEDDING_SERVER_URL = "http://localhost:8000" // Changed from 9000 to 8000
	CHROMA_API_URL       = "http://localhost:8000/api/v1/query"
	CHROMA_HEARTBEAT_URL = "http://localhost:8000/api/v1/heartbeat"
	CHROMA_COLLECTION    = "pubmed-central"
	CHUNK_SIZE           = 150 // Number of words per chunk
	OVERLAP_SIZE         = 15  // Number of overlapping words
)

type ChatRequest struct {
	Message string `json:"message"`
}

type ChatResponse struct {
	Response string `json:"response"`
}

type TextChunk struct {
	Text      string
	StartWord int
	EndWord   int
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

// createTextChunks splits text into overlapping chunks
func createTextChunks(text string) []TextChunk {
	words := strings.Fields(text)
	var chunks []TextChunk

	for i := 0; i < len(words); i += (CHUNK_SIZE - OVERLAP_SIZE) {
		end := i + CHUNK_SIZE
		if end > len(words) {
			end = len(words)
		}

		// Only create chunk if it has meaningful content
		if end-i >= OVERLAP_SIZE {
			chunk := TextChunk{
				Text:      strings.Join(words[i:end], " "),
				StartWord: i,
				EndWord:   end,
			}
			chunks = append(chunks, chunk)
		}

		if end == len(words) {
			break
		}
	}

	return chunks
}

// Helper: Get embedding for a query using Gemini
func getEmbedding(ctx context.Context, text string) ([]float32, error) {
	log.Printf("Getting embedding for text of length: %d", len(text))
	payload := fmt.Sprintf(`{"text": %q}`, text)

	req, err := http.NewRequest("POST", EMBEDDING_SERVER_URL+"/embed", bytes.NewBuffer([]byte(payload)))
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

	// Print embedding details
	log.Printf("Step 2.7: Successfully received embedding from Python server")
	log.Printf("Embedding length: %d", len(pyResp.Embedding))
	if len(pyResp.Embedding) > 10 {
		log.Printf("First 10 values: %v", pyResp.Embedding[:10])
	}

	return pyResp.Embedding, nil
}

// Helper: Query ChromaDB for relevant contexts
func queryChromaDB(queryEmbedding []float32, text string, topK int) ([]string, error) {
	log.Printf("Querying ChromaDB for similar text chunks")

	payload := fmt.Sprintf(`{
        "text": %q,
        "embedding": [%s],
        "n_results": %d
    }`, text, floatArrayToString(queryEmbedding), topK)

	req, err := http.NewRequest("POST", EMBEDDING_SERVER_URL+"/query", bytes.NewBuffer([]byte(payload)))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error querying embedding server: %v", err)
	}
	defer resp.Body.Close()

	var result struct {
		Results []string `json:"results"`
		Count   int      `json:"count"`
	}

	body, _ := ioutil.ReadAll(resp.Body)
	log.Printf("Raw response from embedding server: %s", string(body))

	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&result); err != nil {
		return nil, fmt.Errorf("error decoding response: %v, body: %s", err, string(body))
	}

	if result.Count < topK {
		log.Printf("Warning: Found fewer documents (%d) than requested (%d)", result.Count, topK)
	}

	log.Printf("Found %d matching documents", result.Count)
	return result.Results, nil
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
	startTime := time.Now()

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

	// Start measuring retrieval time
	retrievalStart := time.Now()

	// === RAG: Retrieve relevant PubMed Central context ===
	log.Printf("Step 2: Getting embedding for user query")
	embedding, err := getEmbedding(ctx, req.Message)
	var retrievedContexts []string
	if err == nil && embedding != nil {
		log.Printf("Step 3: Querying ChromaDB for relevant contexts")
		retrievedContexts, err = queryChromaDB(embedding, req.Message, 50) // Pass the actual query
		if err != nil {
			log.Printf("ChromaDB retrieval error: %v", err)
		} else {
			log.Printf("ChromaDB returned %d contexts", len(retrievedContexts))
		}
	} else {
		log.Printf("Embedding error: %v", err)
	}

	retrievalTime := time.Since(retrievalStart)
	log.Printf("Retrieval time: %.2f seconds", retrievalTime.Seconds())

	// Compose prompt with retrieved context
	log.Printf("Step 4: Building prompt for Gemini")
	var promptBuilder strings.Builder
	if len(retrievedContexts) > 0 {
		promptBuilder.WriteString("You are a knowledgeable medical assistant. Below are relevant excerpts from medical literature:\n\n")
		for i, ctx := range retrievedContexts {
			promptBuilder.WriteString(fmt.Sprintf("Excerpt %d:\n%s\n\n", i+1, ctx))
		}
		promptBuilder.WriteString("Using the above excerpts and your medical knowledge, please:\n\n")
		promptBuilder.WriteString("### 1. Key Points:\n\n")
		promptBuilder.WriteString("- Point 1\n")
		promptBuilder.WriteString("- Point 2\n")
		promptBuilder.WriteString("- Point 3\n\n")
		promptBuilder.WriteString("### 2. Detailed Analysis:\n\n")
		promptBuilder.WriteString("For each major point:\n")
		promptBuilder.WriteString("- Main finding\n")
		promptBuilder.WriteString("  * Supporting detail\n")
		promptBuilder.WriteString("  * Evidence from excerpts\n\n")
		promptBuilder.WriteString("### 3. Additional Recommendations:\n\n")
		promptBuilder.WriteString("- Recommendation 1\n")
		promptBuilder.WriteString("- Recommendation 2\n")
		promptBuilder.WriteString("- Recommendation 3\n\n")
		promptBuilder.WriteString("Please maintain this exact formatting with proper line breaks and bullet points.\n\n")
		promptBuilder.WriteString("Question: ")
		promptBuilder.WriteString(req.Message)
	} else {
		promptBuilder.WriteString("You are a knowledgeable medical assistant. Please format your response exactly as follows:\n\n")
		promptBuilder.WriteString("### 1. Key Points:\n\n")
		promptBuilder.WriteString("- Point 1\n")
		promptBuilder.WriteString("- Point 2\n")
		promptBuilder.WriteString("- Point 3\n\n")
		promptBuilder.WriteString("### 2. Detailed Analysis:\n\n")
		promptBuilder.WriteString("For each point:\n")
		promptBuilder.WriteString("- Main concept\n")
		promptBuilder.WriteString("  * Supporting evidence\n")
		promptBuilder.WriteString("  * Detailed explanation\n\n")
		promptBuilder.WriteString("### 3. Recommendations:\n\n")
		promptBuilder.WriteString("- Recommendation 1\n")
		promptBuilder.WriteString("- Recommendation 2\n")
		promptBuilder.WriteString("- Recommendation 3\n\n")
		promptBuilder.WriteString("Please maintain this exact formatting with proper line breaks and bullet points.\n\n")
		promptBuilder.WriteString("Question: ")
		promptBuilder.WriteString(req.Message)
	}
	finalPrompt := promptBuilder.String()
	log.Printf("Final prompt sent to Gemini:\n%s", finalPrompt)

	// Start measuring LLM processing time
	llmStart := time.Now()

	// === Gemini LLM Call ===
	log.Printf("Step 5: Sending prompt to Gemini LLM")
	cs := model.StartChat()
	resp, err := sendMessageWithRetry(ctx, cs, finalPrompt)

	llmTime := time.Since(llmStart)
	log.Printf("LLM processing time: %.2f seconds", llmTime.Seconds())

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

	totalTime := time.Since(startTime)
	log.Printf("\n=== Performance Metrics ===")
	log.Printf("Retrieval Time: %.2f seconds", retrievalTime.Seconds())
	log.Printf("LLM Processing Time: %.2f seconds", llmTime.Seconds())
	log.Printf("Total Response Time: %.2f seconds", totalTime.Seconds())
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

// Function to add a document to ChromaDB (you would call this when indexing documents)
func addDocumentToChromaDB(text string) error {
	chunks := createTextChunks(text)
	log.Printf("Created %d chunks from document", len(chunks))

	for _, chunk := range chunks {
		embedding, err := getEmbedding(context.Background(), chunk.Text)
		if err != nil {
			return fmt.Errorf("error getting embedding for chunk: %v", err)
		}

		// Call the endpoint to add chunk to ChromaDB
		payload := map[string]interface{}{
			"text":      chunk.Text,
			"embedding": embedding,
		}

		jsonData, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("error marshaling payload: %v", err)
		}

		resp, err := http.Post(EMBEDDING_SERVER_URL+"/add_documents", "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			return fmt.Errorf("error adding chunk to ChromaDB: %v", err)
		}
		defer resp.Body.Close()
	}

	return nil
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
