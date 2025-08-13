package raink

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pkoukk/tiktoken-go"
)

const (
	idLen        = 8
	minBatchSize = 2
)

/*
When deciding whether a value belongs in Config or Ranker structs, consider the following:
- Does this value change during operation? → Ranker if yes, Config if no
- Should users be able to configure this directly? → Config if yes, Ranker if no
- Is this derived from other configuration? → Usually Ranker
- Does this require initialization or cleanup? → Usually Ranker
- Is this part of the public API? → Config if yes, Ranker if no
*/

type Config struct {
	InitialPrompt   string           `json:"initial_prompt"`
	BatchSize       int              `json:"batch_size"`
	NumRuns         int              `json:"num_runs"`
	OllamaModel     string           `json:"ollama_model"`
	OpenAIModel     openai.ChatModel `json:"openai_model"`
	TokenLimit      int              `json:"token_limit"`
	RefinementRatio float64          `json:"refinement_ratio"`
	OpenAIKey       string           `json:"-"`
	OpenAIAPIURL    string           `json:"-"`
	OllamaAPIURL    string           `json:"-"`
	Encoding        string           `json:"encoding"`
	BatchTokens     int              `json:"batch_tokens"`
	DryRun          bool             `json:"-"`
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.InitialPrompt == "" {
		return fmt.Errorf("initial prompt cannot be empty")
	}
	if c.BatchSize <= 0 {
		return fmt.Errorf("batch size must be greater than 0")
	}
	if c.NumRuns <= 0 {
		return fmt.Errorf("number of runs must be greater than 0")
	}
	if c.TokenLimit <= 0 {
		return fmt.Errorf("token limit must be greater than 0")
	}
	if c.OllamaModel == "" && c.OpenAIAPIURL == "" && c.OpenAIKey == "" {
		return fmt.Errorf("openai key cannot be empty")
	}
	if c.BatchSize < minBatchSize {
		return fmt.Errorf("batch size must be at least %d", minBatchSize)
	}
	return nil
}

type Ranker struct {
	cfg        *Config
	encoding   *tiktoken.Tiktoken
	rng        *rand.Rand
	numBatches int
	round      int
}

// NewRanker creates a new Ranker instance
func NewRanker(config *Config) (*Ranker, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	encoding, err := tiktoken.GetEncoding(config.Encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get tiktoken encoding: %w", err)
	}

	return &Ranker{
		cfg:      config,
		encoding: encoding,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

// adjustBatchSize dynamically adjusts batch size to fit within token limits
func (ranker *Ranker) adjustBatchSize(objects []object, samples int) error {
	// Dynamically adjust batch size upfront.
	for {
		valid := true
		var estTotalTokens int
		var numBatches int

		for i := 0; i < samples; i++ {
			ranker.rng.Shuffle(len(objects), func(i, j int) {
				objects[i], objects[j] = objects[j], objects[i]
			})
			numBatches = max(1, len(objects)/ranker.cfg.BatchSize) // Need at least one batch.
			for j := 0; j < numBatches; j++ {
				batch := objects[j*ranker.cfg.BatchSize : (j+1)*min(len(objects), ranker.cfg.BatchSize)] // Don't index more objects than we have.
				estBatchTokens := ranker.estimateTokens(batch, true)
				estTotalTokens += estBatchTokens
				if estBatchTokens > ranker.cfg.TokenLimit {
					log.Printf("Sample %d: estimated tokens %d > max token threshold %d", i, estBatchTokens, ranker.cfg.TokenLimit)
					ranker.logTokenSizes(batch)
					valid = false
					break
				}
			}
			if !valid {
				break
			}
		}

		if valid {
			avgEstTokens := estTotalTokens / (samples * numBatches)
			avgEstPct := float64(avgEstTokens) / float64(ranker.cfg.TokenLimit) * 100
			log.Printf("Average estimated tokens: %d (%.2f%% of max %d tokens)", avgEstTokens, avgEstPct, ranker.cfg.TokenLimit)
			break
		}
		if ranker.cfg.BatchSize <= minBatchSize {
			return fmt.Errorf("cannot create a valid batch within the token limit")
		}
		ranker.cfg.BatchSize--
		log.Printf("Decreasing batch size to %d", ranker.cfg.BatchSize)
	}
	return nil
}

type object struct {
	// object unique identifier use to identify the object in the final results
	ID string `json:"id"`
	// string value to be ranked
	Value string `json:"value"`
	// the original structured object if we're loading a json file
	Object interface{} `json:"object"`
}

type rankedObject struct {
	Object object
	Score  float64
}

type rankedObjectResponse struct {
	Objects []string `json:"objects" jsonschema_description:"List of ranked object IDs"`
}

type RankedObject struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	// the original structured object if we're loading a json file
	Object   interface{} `json:"object"`
	Score    float64     `json:"score"`
	Exposure int         `json:"exposure"`
	Rank     int         `json:"rank"`
}

func generateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}

var rankedObjectResponseSchema = generateSchema[rankedObjectResponse]()

// shortDeterministicID generates a short deterministic ID from input string
func shortDeterministicID(input string, length int) string {
	// Keep only A-Za-z0-9 from Base64-encoded SHA-256 hash.
	hash := sha256.Sum256([]byte(input))
	base64Encoded := base64.URLEncoding.EncodeToString(hash[:])
	var result strings.Builder
	for _, char := range base64Encoded {
		if (char >= '0' && char <= '9') || (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') {
			result.WriteRune(char)
		}
	}
	filtered := result.String()
	if length > len(filtered) {
		length = len(filtered)
	}
	return filtered[:length]
}

// RankFromFile ranks objects loaded from a file with optional template
func (r *Ranker) RankFromFile(filePath string, templateData string, forceJSON bool) ([]RankedObject, error) {
	objects, err := loadObjectsFromFile(filePath, templateData, forceJSON)
	if err != nil {
		return nil, err
	}

	// check that no object is too large
	for _, obj := range objects {
		tokens := r.estimateTokens([]object{obj}, true)
		if tokens > r.cfg.BatchTokens {
			return nil, fmt.Errorf("object is too large with %d tokens:\n%s", tokens, obj.Value)
		}
	}

	if err := r.adjustBatchSize(objects, 10); err != nil {
		return nil, err
	}

	results := r.rank(objects, 1)

	// Add the rank key to each final result based on its position in the list
	for i := range results {
		results[i].Rank = i + 1
	}

	return results, nil
}

// loadObjectsFromFile loads objects from a file with optional template
func loadObjectsFromFile(filePath string, templateData string, forceJSON bool) (objects []object, err error) {
	var tmpl *template.Template
	if templateData != "" {
		if templateData[0] == '@' {
			content, err := os.ReadFile(templateData[1:])
			if err != nil {
				return nil, fmt.Errorf("failed to read template file %s: %w", templateData[1:], err)
			}
			templateData = string(content)
		}
		if tmpl, err = template.New("raink-item-template").Parse(templateData); err != nil {
			return nil, fmt.Errorf("failed to parse template: %w", err)
		}
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open input file %s: %w", filePath, err)
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	if ext == ".json" || forceJSON {
		// parse the file in an opaque array
		var data []interface{}
		if err := json.NewDecoder(file).Decode(&data); err != nil {
			return nil, fmt.Errorf("failed to decode JSON from %s: %w", filePath, err)
		}

		// iterate over the map and create objects
		for _, value := range data {
			var valueStr string
			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, value); err != nil {
					return nil, fmt.Errorf("failed to execute template: %w", err)
				}
				valueStr = tmplData.String()
			} else {
				log.Printf("WARNING: using json input without a template, using JSON object as it is\n")
				jsonValue, err := json.Marshal(value)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal JSON value: %w", err)
				}
				valueStr = string(jsonValue)
			}

			id := shortDeterministicID(valueStr, idLen)
			objects = append(objects, object{ID: id, Object: value, Value: valueStr})
		}
	} else {
		// read and interpolate the file line by line
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("failed to read line from %s: %w", filePath, err)
			}
			line = strings.TrimSpace(line)

			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, map[string]string{"Data": line}); err != nil {
					return nil, fmt.Errorf("failed to execute template on line: %w", err)
				}
				line = tmplData.String()
			}

			id := shortDeterministicID(line, idLen)
			objects = append(objects, object{ID: id, Object: nil, Value: line})
		}
	}

	return objects, nil
}

// rank performs the ranking algorithm on the given objects
func (r *Ranker) rank(objects []object, round int) []RankedObject {
	r.round = round

	log.Printf("Round %d: Ranking %d objects\n", r.round, len(objects))

	// If we've narrowed down to a single object, we're done.
	if len(objects) == 1 {
		return []RankedObject{
			{
				Key:      objects[0].ID,
				Value:    objects[0].Value,
				Object:   objects[0].Object,
				Score:    0, // 0 is guaranteed to be the "highest" score.
				Exposure: 1,
			},
		}
	}

	// Downstream ranking gets unhappy if we try to rank more objects than we
	// have.
	if r.cfg.BatchSize > len(objects) {
		r.cfg.BatchSize = len(objects)
	}

	r.numBatches = len(objects) / r.cfg.BatchSize

	// Process the objects and get the sorted results.
	results := r.shuffleBatchRank(objects)

	// If the refinement ratio is 0, that effectively means we're refining
	// _none_ of the top objects, so we're done.
	if r.cfg.RefinementRatio == 0 {
		return results
	}

	// Calculate the mid index based on the refinement ratio.
	mid := int(float64(len(results)) * r.cfg.RefinementRatio)
	topPortion := results[:mid]
	bottomPortion := results[mid:]

	// If we haven't reduced the number of objects (as may eventually happen
	// for a ratio above 0.5), we're done.
	if len(topPortion) == len(objects) {
		return results
	}

	log.Println("Top items being sent back into recursion:")
	for i, obj := range topPortion {
		log.Printf("Rank %d: ID=%s, Score=%.2f, Value=%s", i+1, obj.Key, obj.Score, obj.Value)
	}

	var topPortionObjects []object
	for _, result := range topPortion {
		topPortionObjects = append(topPortionObjects, object{ID: result.Key, Value: result.Value, Object: result.Object})
	}

	refinedTopPortion := r.rank(topPortionObjects, round+1)

	// Adjust scores by recursion depth; this serves as an inverted weight so
	// that later rounds are guaranteed to sit higher in the final list.
	for i := range refinedTopPortion {
		refinedTopPortion[i].Score /= float64(2 * round)
	}

	// Combine the refined top portion with the unrefined bottom portion.
	finalResults := append(refinedTopPortion, bottomPortion...)

	return finalResults
}

func (r *Ranker) logFromApiCall(runNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf("Round %d, Run %*d/%d, Batch %*d/%d: "+message, r.round, len(strconv.Itoa(r.cfg.NumRuns)), runNum, r.cfg.NumRuns, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches)
	log.Printf(formattedMessage, args...)
}

func (r *Ranker) shuffleBatchRank(objects []object) []RankedObject {
	scores := make(map[string][]float64)

	exposureCounts := make(map[string]int)

	type batchResult struct {
		rankedObjects []rankedObject
		err           error
	}
	resultsChan := make(chan batchResult, r.numBatches)

	var firstRunRemainderItems []object

	for i := 0; i < r.cfg.NumRuns; i++ {
		r.rng.Shuffle(len(objects), func(i, j int) {
			objects[i], objects[j] = objects[j], objects[i]
		})

		// Ensure remainder items from the first run are not in the remainder
		// range in the second run
		if i == 1 && len(firstRunRemainderItems) > 0 {
			for {
				remainderStart := r.numBatches * r.cfg.BatchSize
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
				r.rng.Shuffle(len(objects), func(i, j int) {
					objects[i], objects[j] = objects[j], objects[i]
				})
			}
		}

		// Split into groups of batchSize and process them concurrently
		log.Printf("Round %d, Run %*d/%d: Submitting batches to API\n", r.round, len(strconv.Itoa(r.cfg.NumRuns)), i+1, r.cfg.NumRuns)
		for j := 0; j < r.numBatches; j++ {
			batch := objects[j*r.cfg.BatchSize : (j+1)*r.cfg.BatchSize]
			go func(runNumber, batchNumber int, batch []object) {
				rankedBatch, err := r.rankObjects(batch, runNumber, batchNumber)
				resultsChan <- batchResult{rankedObjects: rankedBatch, err: err}
			}(i+1, j+1, batch)
		}

		// Collect results from all batches
		for j := 0; j < r.numBatches; j++ {
			result := <-resultsChan
			if result.err != nil {
				log.Printf("Error in batch processing: %v", result.err)
				continue // Skip this batch but continue with others
			}
			for _, rankedObject := range result.rankedObjects {
				scores[rankedObject.Object.ID] = append(scores[rankedObject.Object.ID], rankedObject.Score)
				exposureCounts[rankedObject.Object.ID]++ // Update exposure count
			}
		}

		// Save remainder items from the first run
		if i == 0 {
			remainderStart := r.numBatches * r.cfg.BatchSize
			if remainderStart < len(objects) {
				firstRunRemainderItems = make([]object, len(objects[remainderStart:]))
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

	var results []RankedObject
	for id, score := range finalScores {
		for _, obj := range objects {
			if obj.ID == id {
				results = append(results, RankedObject{
					Key:      id,
					Value:    obj.Value,
					Object:   obj.Object,
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

func (r *Ranker) logTokenSizes(group []object) {
	log.Println("Logging token sizes for each object in the batch:")
	for _, obj := range group {
		tokenSize := r.estimateTokens([]object{obj}, false)
		valuePreview := obj.Value
		if len(valuePreview) > 100 {
			valuePreview = valuePreview[:100]
		}
		log.Printf("Object ID: %s, Token Size: %d, Value Preview: %s", obj.ID, tokenSize, valuePreview)
	}
}

const promptFmt = "id: `%s`\nvalue:\n```\n%s\n```\n\n"

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

func (r *Ranker) estimateTokens(group []object, includePrompt bool) int {
	text := ""
	if includePrompt {
		text += r.cfg.InitialPrompt + promptDisclaimer
	}
	for _, obj := range group {
		text += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}

	if r.cfg.OllamaModel != "" {
		// TODO: Update to use Ollama tokenize API when this PR is merged:
		// https://github.com/ollama/ollama/pull/6586
		return len(text) / 4
	} else {
		return len(r.encoding.Encode(text, nil, nil))
	}
}

func (r *Ranker) rankObjects(group []object, runNumber int, batchNumber int) ([]rankedObject, error) {
	prompt := r.cfg.InitialPrompt + promptDisclaimer
	for _, obj := range group {
		prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}

	if r.cfg.DryRun {
		log.Printf("Dry run API call")
		// Simulate a ranked response for dry run
		var rankedObjects []rankedObject
		for i, obj := range group {
			rankedObjects = append(rankedObjects, rankedObject{
				Object: obj,
				Score:  float64(i + 1), // Simulate scores based on position
			})
		}
		return rankedObjects, nil
	}

	var rankedResponse rankedObjectResponse
	inputIDs := make(map[string]bool)
	for _, obj := range group {
		inputIDs[obj.ID] = true
	}
	var err error
	if r.cfg.OllamaModel != "" {
		rankedResponse, err = r.callOllama(prompt, runNumber, batchNumber, inputIDs)
	} else {
		rankedResponse, err = r.callOpenAI(prompt, runNumber, batchNumber, inputIDs)
	}
	if err != nil {
		return nil, err
	}

	// Assign scores based on position in the ranked list
	var rankedObjects []rankedObject
	for i, id := range rankedResponse.Objects {
		for _, obj := range group {
			if obj.ID == id {
				rankedObjects = append(rankedObjects, rankedObject{
					Object: obj,
					Score:  float64(i + 1), // Score based on position (1 for first, 2 for second, etc.)
				})
				break
			}
		}
	}

	return rankedObjects, nil
}

type customTransport struct {
	Transport  http.RoundTripper
	Headers    http.Header
	StatusCode int
	Body       []byte
}

func (t *customTransport) RoundTrip(req *http.Request) (*http.Response, error) {
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

// validateIDs updates the rankedResponse in place to fix case-insensitive ID mismatches.
// If any IDs are missing, returns the missing IDs along with an error.
func validateIDs(rankedResponse *rankedObjectResponse, inputIDs map[string]bool) ([]string, error) {
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

func (r *Ranker) callOpenAI(prompt string, runNum int, batchNum int, inputIDs map[string]bool) (rankedObjectResponse, error) {

	customTransport := &customTransport{Transport: http.DefaultTransport}
	customClient := &http.Client{Transport: customTransport}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(r.cfg.OpenAIKey),
		option.WithHTTPClient(customClient),
		option.WithMaxRetries(5),
	}

	// Add base URL option if specified
	if r.cfg.OpenAIAPIURL != "" {
		// Ensure the URL ends with a trailing slash
		baseURL := r.cfg.OpenAIAPIURL
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(clientOptions...)

	backoff := time.Second

	conversationHistory := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(prompt),
	}

	var rankedResponse rankedObjectResponse
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
						Schema:      openai.F(rankedObjectResponseSchema),
						Strict:      openai.Bool(true),
					}),
				},
			),
			Model: openai.F(r.cfg.OpenAIModel),
		})
		if err == nil {

			conversationHistory = append(conversationHistory,
				openai.AssistantMessage(completion.Choices[0].Message.Content),
			)

			err = json.Unmarshal([]byte(completion.Choices[0].Message.Content), &rankedResponse)
			if err != nil {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Error unmarshalling response: %v\n", err))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(invalidJSONStr),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				log.Printf("OpenAI API response: %s", trimmedContent)
				continue
			}

			missingIDs, err := validateIDs(&rankedResponse, inputIDs)
			if err != nil {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "))),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				log.Printf("OpenAI API response: %s", trimmedContent)
				continue
			}

			return rankedResponse, nil
		}

		if err == context.DeadlineExceeded {
			r.logFromApiCall(runNum, batchNum, "Context deadline exceeded, retrying...")
			time.Sleep(backoff)
			backoff *= 2
			continue
		}

		if customTransport.StatusCode == http.StatusTooManyRequests {
			for key, values := range customTransport.Headers {
				if strings.HasPrefix(key, "X-Ratelimit") {
					for _, value := range values {
						r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Rate limit header: %s: %s", key, value))
					}
				}
			}

			respBody := customTransport.Body
			if respBody == nil {
				r.logFromApiCall(runNum, batchNum, "Error reading response body: %v", "response body is nil")
			} else {
				r.logFromApiCall(runNum, batchNum, "Response body: %s", string(respBody))
			}

			remainingTokensStr := customTransport.Headers.Get("X-Ratelimit-Remaining-Tokens")
			resetTokensStr := customTransport.Headers.Get("X-Ratelimit-Reset-Tokens")

			remainingTokens, _ := strconv.Atoi(remainingTokensStr)
			resetDuration, _ := time.ParseDuration(strings.Replace(resetTokensStr, "s", "s", 1))

			r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Rate limit exceeded. Suggested wait time: %v. Remaining tokens: %d", resetDuration, remainingTokens))

			if resetDuration > 0 {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Waiting for %v before retrying...", resetDuration))
				time.Sleep(resetDuration)
			} else {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Waiting for %v before retrying...", backoff))
				time.Sleep(backoff)
				backoff *= 2
			}
		} else {
			return rankedObjectResponse{}, fmt.Errorf("run %*d/%d, batch %*d/%d: unexpected error: %w", len(strconv.Itoa(r.cfg.NumRuns)), runNum, r.cfg.NumRuns, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches, err)
		}
	}
}

func (r *Ranker) callOllama(prompt string, runNum int, batchNum int, inputIDs map[string]bool) (rankedObjectResponse, error) {

	var rankedResponse rankedObjectResponse

	// Initialize the conversation history with the initial prompt
	conversationHistory := []map[string]interface{}{
		{"role": "user", "content": prompt},
	}

	for {

		requestBody, err := json.Marshal(map[string]interface{}{
			"model":    r.cfg.OllamaModel,
			"stream":   false,
			"format":   "json",
			"num_ctx":  r.cfg.BatchTokens,
			"messages": conversationHistory,
		})
		if err != nil {
			return rankedObjectResponse{}, fmt.Errorf("error creating Ollama API request body: %w", err)
		}

		req, err := http.NewRequest("POST", r.cfg.OllamaAPIURL, bytes.NewReader(requestBody))
		if err != nil {
			return rankedObjectResponse{}, fmt.Errorf("error creating Ollama API request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}

		resp, err := client.Do(req)
		if err != nil {
			return rankedObjectResponse{}, fmt.Errorf("error making request to Ollama API: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			return rankedObjectResponse{}, fmt.Errorf("Ollama API returned an error: %v, body: %s", resp.StatusCode, body)
		}

		responseBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return rankedObjectResponse{}, fmt.Errorf("error reading Ollama API response body: %w", err)
		}

		var ollamaResponse struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}

		err = json.Unmarshal(responseBody, &ollamaResponse)
		if err != nil {
			return rankedObjectResponse{}, fmt.Errorf("error parsing Ollama API response: %w", err)
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
			r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Error unmarshalling response: %v\n", err))
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
			r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
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

		return rankedResponse, nil
	}
}
