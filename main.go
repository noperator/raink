package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkoukk/tiktoken-go"
)

// TODO: Move these attributes to CLI args.

// https://platform.openai.com/docs/models/gp#models-overview
const maxTokens = 128000
const tokenLimitThreshold = 0.95 * maxTokens

// https://pkg.go.dev/github.com/openai/openai-go@v0.1.0-alpha.38#ChatModel
// const model = openai.ChatModelGPT4o2024_08_06
const model = openai.ChatModelGPT4oMini

// https://github.com/pkoukk/tiktoken-go?tab=readme-ov-file#available-models
const encoding = "o200k_base"

const idLen = 8

type Object struct {
	ID    string `json:"id"`
	Value string `json:"value"`
}

type RankedObject struct {
	Object Object
	Score  float64
}

type RankedObjectResponse struct {
	Objects []string `json:"objects" jsonschema_description:"List of ranked object IDs"`
}

type FinalResult struct {
	Key      string  `json:"key"`
	Value    string  `json:"value"`
	Score    float64 `json:"score"`
	Exposure int     `json:"exposure"`
	Rank     int     `json:"rank"`
}

var dryRun bool

func GenerateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}

var RankedObjectResponseSchema = GenerateSchema[RankedObjectResponse]()

func ShortDeterministicID(input string, length int) string {
	// Step 1: Hash the input using SHA-256
	hash := sha256.Sum256([]byte(input))

	// Step 2: Encode the hash in Base64 (URL-safe)
	base64Encoded := base64.URLEncoding.EncodeToString(hash[:])

	// Step 3: Truncate to the desired length
	if length > len(base64Encoded) {
		length = len(base64Encoded) // Avoid out-of-bounds access
	}
	return base64Encoded[:length]
}

func main() {
	log.SetOutput(os.Stderr)

	inputFile := flag.String("f", "", "Input file")
	batchSize := flag.Int("s", 10, "Batch size")
	numRuns := flag.Int("r", 10, "Number of runs")
	initialPrompt := flag.String("p", "", "Initial prompt")
	ollamaModel := flag.String("ollama-model", "", "Ollama model name (if not set, OpenAI will be used)")
	flag.BoolVar(&dryRun, "dry-run", false, "Enable dry run mode (log API calls without making them)")
	flag.Parse()

	if *inputFile == "" {
		log.Println("Usage: go run main.go -f <input_file> [-s <batch_size>] [-r <num_runs>] [-p <initial_prompt>] [--ollama-model <model_name>]")
		return
	}

	file, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var objects []Object
	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatal(err)
		}
		line = strings.TrimSpace(line)
		id := ShortDeterministicID(line, idLen)
		objects = append(objects, Object{ID: id, Value: line})
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	encoding, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		log.Fatal("Failed to get tiktoken encoding:", err)
	}

	// Adjust batch size upfront
	currentBatchSize := *batchSize
	for {
		valid := true
		var totalTokens int
		var totalBatches int

		for i := 0; i < 10; i++ {
			rng.Shuffle(len(objects), func(i, j int) {
				objects[i], objects[j] = objects[j], objects[i]
			})
			log.Printf("Estimating tokens for batch size %d with object count %d", currentBatchSize, len(objects))
			totalBatches = len(objects) / currentBatchSize
			for j := 0; j < totalBatches; j++ {
				group := objects[j*currentBatchSize : (j+1)*currentBatchSize]
				est := estimateTokens(group, *initialPrompt, encoding)
				totalTokens += est
				if est > tokenLimitThreshold {
					log.Printf("shuffle %d: Estimated tokens %d > max token threshold %f", i, est, tokenLimitThreshold)
					logTokenSizes(group, *initialPrompt, encoding)
					valid = false
					break
				}
			}
			if !valid {
				break
			}
		}

		if totalBatches > 0 {
			averageTokens := totalTokens / totalBatches
			averagePercentage := float64(averageTokens) / maxTokens * 100
			log.Printf("Average estimated tokens: %d (%.2f%% of max tokens)", averageTokens, averagePercentage)
		}

		if valid {
			break
		}
		currentBatchSize--
		log.Printf("Decreasing batch size to %d", currentBatchSize)
		if currentBatchSize == 0 {
			log.Fatal("Cannot create a valid batch within the token limit")
		}
	}

	// Recursive processing
	finalResults := recursiveProcess(objects, currentBatchSize, *numRuns, *initialPrompt, rng, 1, ollamaModel)

	// Add the rank key to each final result based on its position in the list
	for i := range finalResults {
		finalResults[i].Rank = i + 1
	}

	jsonResults, err := json.MarshalIndent(finalResults, "", "  ")
	if err != nil {
		panic(err)
	}

	if !dryRun {
		fmt.Println(string(jsonResults))
	}
}

func recursiveProcess(objects []Object, batchSize, numRuns int, initialPrompt string, rng *rand.Rand, depth int, ollamaModel *string) []FinalResult {
	// If we have only one object, return it with the highest score
	if len(objects) == 1 {
		return []FinalResult{
			{
				Key:      objects[0].ID,
				Value:    objects[0].Value,
				Score:    0, // Set score to 0 to guarantee it's the "highest" score
				Exposure: 1, // Since it's the only one, it has been exposed once
			},
		}
	}

	if batchSize > len(objects) {
		batchSize = len(objects)
	}

	// Process the objects and get the sorted results
	results := processObjects(objects, batchSize, numRuns, initialPrompt, rng, ollamaModel)

	// TODO: Move this ratio (50%) to a CLI arg.
	mid := len(results) / 2
	topHalf := results[:mid]
	bottomHalf := results[mid:]

	log.Println("Top items being sent back into recursion:")
	for i, obj := range topHalf {
		log.Printf("Rank %d: ID=%s, Score=%.2f, Value=%s", i+1, obj.Key, obj.Score, obj.Value)
	}

	var topHalfObjects []Object
	for _, result := range topHalf {
		topHalfObjects = append(topHalfObjects, Object{ID: result.Key, Value: result.Value})
	}

	refinedTopHalf := recursiveProcess(topHalfObjects, batchSize, numRuns, initialPrompt, rng, depth+1, ollamaModel)

	// Adjust scores by recursion depth
	for i := range refinedTopHalf {
		refinedTopHalf[i].Score /= float64(2 * depth)
	}

	// Combine the refined top half with the unrefined bottom half
	finalResults := append(refinedTopHalf, bottomHalf...)

	return finalResults
}

// TODO: Also log the "round" number (i.e., the repeated recursion depth).
func logRunBatch(runNumber, totalRuns, batchNumber, totalBatches int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf("Run %*d/%d, Batch %*d/%d: "+message, len(strconv.Itoa(totalRuns)), runNumber, totalRuns, len(strconv.Itoa(totalBatches)), batchNumber, totalBatches)
	log.Printf(formattedMessage, args...)
}

func processObjects(objects []Object, batchSize, numRuns int, initialPrompt string, rng *rand.Rand, ollamaModel *string) []FinalResult {
	scores := make(map[string][]float64)

	totalBatches := len(objects) / batchSize

	exposureCounts := make(map[string]int)

	resultsChan := make(chan []RankedObject, totalBatches)

	var firstRunRemainderItems []Object

	for i := 0; i < numRuns; i++ {
		rng.Shuffle(len(objects), func(i, j int) {
			objects[i], objects[j] = objects[j], objects[i]
		})

		// Ensure remainder items from the first run are not in the remainder range in the second run
		if i == 1 && len(firstRunRemainderItems) > 0 {
			for {
				remainderStart := totalBatches * batchSize
				remainderItems := objects[remainderStart:]
				conflictFound := false
				for _, item := range remainderItems {
					for _, firstRunItem := range firstRunRemainderItems {
						if item.ID == firstRunItem.ID {
							log.Printf("Conflicting remainder item found: %v, %v\n", item, firstRunItem)
							conflictFound = true
							break
						}
					}
					if conflictFound {
						break
					}
				}
				if !conflictFound {
					break
				}
				rng.Shuffle(len(objects), func(i, j int) {
					objects[i], objects[j] = objects[j], objects[i]
				})
			}
		}

		// Split into groups of batchSize and process them concurrently
		log.Printf("Run %*d/%d: Submitting batches to API\n", len(strconv.Itoa(numRuns)), i+1, numRuns)
		for j := 0; j < totalBatches; j++ {
			group := objects[j*batchSize : (j+1)*batchSize]
			go func(runNumber, batchNumber int, group []Object) {
				// formattedMessage := fmt.Sprintf("Run %*d/%d, Batch %*d/%d: Submitting batch to API\n", len(strconv.Itoa(numRuns)), runNumber, numRuns, len(strconv.Itoa(totalBatches)), batchNumber, totalBatches)
				// log.Printf(formattedMessage)
				rankedGroup := rankGroup(group, runNumber, numRuns, batchNumber, totalBatches, initialPrompt, ollamaModel)
				resultsChan <- rankedGroup
			}(i+1, j+1, group)
		}

		// Collect results from all batches
		for j := 0; j < totalBatches; j++ {
			rankedGroup := <-resultsChan
			for _, rankedObject := range rankedGroup {
				scores[rankedObject.Object.ID] = append(scores[rankedObject.Object.ID], rankedObject.Score)
				exposureCounts[rankedObject.Object.ID]++ // Update exposure count
			}
		}

		// Save remainder items from the first run
		if i == 0 {
			remainderStart := totalBatches * batchSize
			if remainderStart < len(objects) {
				firstRunRemainderItems = make([]Object, len(objects[remainderStart:]))
				copy(firstRunRemainderItems, objects[remainderStart:])
				log.Printf("First run remainder items: %v\n", firstRunRemainderItems)
			}
		}
	}

	// Calculate average scores
	finalScores := make(map[string]float64)
	for id, scoreList := range scores {
		var sum float64
		for _, score := range scoreList {
			sum += score
		}
		finalScores[id] = sum / float64(len(scoreList))
	}

	var results []FinalResult
	for id, score := range finalScores {
		for _, obj := range objects {
			if obj.ID == id {
				results = append(results, FinalResult{
					Key:      id,
					Value:    obj.Value,
					Score:    score,
					Exposure: exposureCounts[id], // Include exposure count
				})
				break
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	return results
}

func logTokenSizes(group []Object, initialPrompt string, encoding *tiktoken.Tiktoken) {
	log.Println("Logging token sizes for each object in the batch:")
	for _, obj := range group {
		tokenSize := estimateTokens([]Object{obj}, initialPrompt, encoding)
		valuePreview := obj.Value
		if len(valuePreview) > 100 {
			valuePreview = valuePreview[:100]
		}
		log.Printf("Object ID: %s, Token Size: %d, Value Preview: %s", obj.ID, tokenSize, valuePreview)
	}
}

const promptFmt = "id: `%s`\nvalue:\n```\n%s\n```\n\n"

// TODO: Merge these and clean them up.

var promptDisclaimer = fmt.Sprintf(
	"\n\nREMEMBER to:\n"+
		"- ALWAYS respond with the short %d-character ID of each item found above the value "+
		"(i.e., I'll provide you with `id: <ID>` above the value, and you should respond with that same ID in your response)\n"+
		"— NEVER respond with the actual value!\n"+
		"— NEVER include backticks around IDs in your response!\n"+
		"— NEVER include scores or a written reason/justification in your response!\n"+
		"- Respond in RANKED DESCENDING order, where the FIRST item in your response is the MOST RELEVANT\n"+
		"- Respond in JSON format, with the following schema:\n  {\"objects\": [\"<ID1>\", \"<ID2>\", ...]}\n\n"+
		"Here are the objects to be ranked:\n\n",
	idLen,
)

const missingIDsStr = "Your last response was missing the following IDs: [%s]. " +
	"Try again—and make ABSOLUTELY SURE to remember to:\n" +
	"- ALWAYS return the IDs and NOT THE VALUES! " +
	"- ALWAYS respond in JSON format as specified! " +
	"- ALWAYS return ALL of the IDs in the list!" +
	"- NEVER include backticks around IDs in your response!" +
	"— NEVER include scores or a written reason/justification in your response!"

const invalidJSONStr = "Your last response was not valid JSON. Try again!"

func estimateTokens(group []Object, initialPrompt string, encoding *tiktoken.Tiktoken) int {
	prompt := initialPrompt + promptDisclaimer
	for _, obj := range group {
		prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}
	return len(encoding.Encode(prompt, nil, nil))
}

func rankGroup(group []Object, runNumber int, totalRuns int, batchNumber int, totalBatches int, initialPrompt string, ollamaModel *string) []RankedObject {
	prompt := initialPrompt + promptDisclaimer
	for _, obj := range group {
		prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}

	if dryRun {
		log.Printf("Dry run API call")
		// Simulate a ranked response for dry run
		var rankedObjects []RankedObject
		for i, obj := range group {
			rankedObjects = append(rankedObjects, RankedObject{
				Object: obj,
				Score:  float64(i + 1), // Simulate scores based on position
			})
		}
		return rankedObjects
	}

	var rankedResponse RankedObjectResponse
	inputIDs := make(map[string]bool)
	for _, obj := range group {
		inputIDs[obj.ID] = true
	}
	if ollamaModel != nil && *ollamaModel != "" {
		rankedResponse = callOllama(prompt, *ollamaModel, runNumber, totalRuns, batchNumber, totalBatches, inputIDs)
	} else {
		rankedResponse = callOpenAI(prompt, runNumber, totalRuns, batchNumber, totalBatches, inputIDs)
	}

	// Assign scores based on position in the ranked list
	var rankedObjects []RankedObject
	for i, id := range rankedResponse.Objects {
		for _, obj := range group {
			if obj.ID == id {
				rankedObjects = append(rankedObjects, RankedObject{
					Object: obj,
					Score:  float64(i + 1), // Score based on position (1 for first, 2 for second, etc.)
				})
				break
			}
		}
	}

	return rankedObjects
}

type CustomTransport struct {
	Transport  http.RoundTripper
	Headers    http.Header
	StatusCode int
	Body       []byte
}

func (t *CustomTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.Transport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	t.Headers = resp.Header
	t.StatusCode = resp.StatusCode

	t.Body, err = io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	resp.Body = io.NopCloser(bytes.NewBuffer(t.Body))

	return resp, nil
}

// Updates the rankedResponse in place to fix case-insensitive ID mismatches. If any IDs are missing, returns the missing IDs along with an error.
func validateIDs(rankedResponse *RankedObjectResponse, inputIDs map[string]bool) ([]string, error) {
	// Create a map for case-insensitive ID matching
	inputIDsLower := make(map[string]string)
	for id := range inputIDs {
		inputIDsLower[strings.ToLower(id)] = id
	}

	missingIDs := make(map[string]bool)
	for id := range inputIDs {
		missingIDs[id] = true
	}

	for i, id := range rankedResponse.Objects {
		id = strings.ReplaceAll(id, "`", "")
		lowerID := strings.ToLower(id)
		if correctID, found := inputIDsLower[lowerID]; found {
			if correctID != id {
				// Replace the case-wrong match with the correct ID
				rankedResponse.Objects[i] = correctID
			}
			delete(missingIDs, correctID)
		}
	}

	if len(missingIDs) == 0 {
		return nil, nil
	} else {
		missingIDsKeys := make([]string, 0, len(missingIDs))
		for id := range missingIDs {
			missingIDsKeys = append(missingIDsKeys, id)
		}
		return missingIDsKeys, fmt.Errorf("missing IDs: %s", strings.Join(missingIDsKeys, ", "))
	}
}

func callOpenAI(prompt string, runNumber int, totalRuns int, batchNumber int, totalBatches int, inputIDs map[string]bool) RankedObjectResponse {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	customTransport := &CustomTransport{Transport: http.DefaultTransport}
	customClient := &http.Client{Transport: customTransport}

	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(customClient),
		option.WithMaxRetries(5),
	)

	backoff := time.Second

	conversationHistory := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(prompt),
	}

	var rankedResponse RankedObjectResponse
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Messages: openai.F(conversationHistory),
			ResponseFormat: openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
				openai.ResponseFormatJSONSchemaParam{
					Type: openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
					JSONSchema: openai.F(openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        openai.F("ranked_object_response"),
						Description: openai.F("List of ranked object IDs"),
						Schema:      openai.F(RankedObjectResponseSchema),
						Strict:      openai.Bool(true),
					}),
				},
			),
			Model: openai.F(model),
		})
		if err == nil {

			conversationHistory = append(conversationHistory,
				openai.AssistantMessage(completion.Choices[0].Message.Content),
			)

			err = json.Unmarshal([]byte(completion.Choices[0].Message.Content), &rankedResponse)
			if err != nil {
				logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, fmt.Sprintf("Error unmarshalling response: %v\n", err))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(invalidJSONStr),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				log.Printf("Ollama API response: %s", trimmedContent)
				continue
			}

			missingIDs, err := validateIDs(&rankedResponse, inputIDs)
			if err != nil {
				logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "))),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				log.Printf("Ollama API response: %s", trimmedContent)
				continue
			}

			return rankedResponse
		}

		if err == context.DeadlineExceeded {
			logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, "Context deadline exceeded, retrying...")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		if customTransport.StatusCode == http.StatusTooManyRequests {
			for key, values := range customTransport.Headers {
				if strings.HasPrefix(key, "X-Ratelimit") {
					for _, value := range values {
						log.Printf("Run %d/%d, Batch %d/%d: Rate limit header: %s: %s", runNumber, totalRuns, batchNumber, totalBatches, key, value)
					}
				}
			}

			respBody := customTransport.Body
			if respBody == nil {
				log.Printf("Run %d/%d, Batch %d/%d: Error reading response body: %v", runNumber, totalRuns, batchNumber, totalBatches, "response body is nil")
			} else {
				log.Printf("Run %d/%d, Batch %d/%d: Response body: %s", runNumber, totalRuns, batchNumber, totalBatches, string(respBody))
			}

			remainingTokensStr := customTransport.Headers.Get("X-Ratelimit-Remaining-Tokens")
			resetTokensStr := customTransport.Headers.Get("X-Ratelimit-Reset-Tokens")

			remainingTokens, _ := strconv.Atoi(remainingTokensStr)
			resetDuration, _ := time.ParseDuration(strings.Replace(resetTokensStr, "s", "s", 1))

			logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, "Rate limit exceeded. Suggested wait time: %v. Remaining tokens: %d", resetDuration, remainingTokens)

			if resetDuration > 0 {
				logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, "Waiting for %v before retrying...", resetDuration)
				time.Sleep(resetDuration)
			} else {
				logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, "Waiting for %v before retrying...", backoff)
				time.Sleep(backoff)
				backoff *= 2
			}
		} else {
			log.Fatalf("Run %*d/%d, Batch %*d/%d: Unexpected error: %v", len(strconv.Itoa(totalRuns)), runNumber, totalRuns, len(strconv.Itoa(totalBatches)), batchNumber, totalBatches, err)
		}
	}
}

func callOllama(prompt string, model string, runNumber int, totalRuns int, batchNumber int, totalBatches int, inputIDs map[string]bool) RankedObjectResponse {
	apiURL := os.Getenv("OLLAMA_API_URL")
	if apiURL == "" {
		apiURL = "http://localhost:11434/api/chat"
	}

	var rankedResponse RankedObjectResponse

	// Initialize the conversation history with the initial prompt
	conversationHistory := []map[string]interface{}{
		{"role": "user", "content": prompt},
	}

	for {

		requestBody, err := json.Marshal(map[string]interface{}{
			"model":    model,
			"stream":   false,
			"format":   "json",
			"messages": conversationHistory,
		})
		if err != nil {
			log.Fatalf("Error creating Ollama API request body: %v", err)
		}

		req, err := http.NewRequest("POST", apiURL, bytes.NewReader(requestBody))
		if err != nil {
			log.Fatalf("Error creating Ollama API request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}

		resp, err := client.Do(req)
		if err != nil {
			log.Fatalf("Error making request to Ollama API: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			log.Fatalf("Ollama API returned an error: %v, body: %s", resp.StatusCode, body)
		}

		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Fatalf("Error reading Ollama API response body: %v", err)
		}

		var ollamaResponse struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}

		err = json.Unmarshal(responseBody, &ollamaResponse)
		if err != nil {
			log.Fatalf("Error parsing Ollama API response: %v", err)
		}

		conversationHistory = append(
			conversationHistory,
			map[string]interface{}{
				"role":    "assistant",
				"content": ollamaResponse.Message.Content,
			},
		)

		err = json.Unmarshal([]byte(ollamaResponse.Message.Content), &rankedResponse)
		if err != nil {
			logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, fmt.Sprintf("Error unmarshalling response: %v\n", err))
			conversationHistory = append(conversationHistory,
				map[string]interface{}{
					"role":    "user",
					"content": invalidJSONStr,
				},
			)
			trimmedContent := strings.TrimSpace(ollamaResponse.Message.Content)
			log.Printf("Ollama API response: %s", trimmedContent)
			continue
		}

		missingIDs, err := validateIDs(&rankedResponse, inputIDs)
		if err != nil {
			logRunBatch(runNumber, totalRuns, batchNumber, totalBatches, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
			conversationHistory = append(conversationHistory,
				map[string]interface{}{
					"role":    "user",
					"content": fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", ")),
				},
			)
			trimmedContent := strings.TrimSpace(ollamaResponse.Message.Content)
			log.Printf("Ollama API response: %s", trimmedContent)
			continue
		}

		return rankedResponse
	}
}
