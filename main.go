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

const idLen = 8

/*
When deciding whether a value belongs in Config or Ranker structs, consider the following:
- Does this value change during operation? → Ranker if yes, Config if no
- Should users be able to configure this directly? → Config if yes, Ranker if no
- Is this derived from other configuration? → Usually Ranker
- Does this require initialization or cleanup? → Usually Ranker
- Is this part of the public API? → Config if yes, Ranker if no
*/

type Config struct {
	InitialPrompt   string
	BatchSize       int
	NumRuns         int
	OllamaModel     string
	OpenAIModel     openai.ChatModel
	TokenLimit      int
	RefinementRatio float64
	OpenAIKey       string
	OllamaAPIURL    string
	Encoding        string
	BatchTokens     int
	DryRun          bool
}

// TODO: Move all CLI flag validation this func instead.
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
	if c.OllamaModel == "" && c.OpenAIKey == "" {
		return fmt.Errorf("openai key cannot be empty")
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

func (ranker *Ranker) AdjustBatchSize(objects []Object, samples int) {
	// Dynamically adjust batch size upfront.
	for {
		valid := true
		var estTotalTokens int
		var numBatches int

		for i := 0; i < samples; i++ {
			ranker.rng.Shuffle(len(objects), func(i, j int) {
				objects[i], objects[j] = objects[j], objects[i]
			})
			numBatches = len(objects) / ranker.cfg.BatchSize
			for j := 0; j < numBatches; j++ {
				batch := objects[j*ranker.cfg.BatchSize : (j+1)*ranker.cfg.BatchSize]
				estBatchTokens := ranker.estimateTokens(batch)
				estTotalTokens += estBatchTokens
				if estBatchTokens > ranker.cfg.TokenLimit {
					log.Printf("Sample %d: estimated tokens %d > max token threshold %d", i, estBatchTokens, ranker.cfg.TokenLimit)
					ranker.logTokenSizes(batch)
					break
				}
			}
			if !valid {
				break
			}
		}

		if numBatches > 0 {
			avgEstTokens := estTotalTokens / (samples * numBatches)
			avgEstPct := float64(avgEstTokens) / float64(ranker.cfg.BatchTokens) * 100
			log.Printf("Average estimated tokens: %d (%.2f%% of max %d tokens)", avgEstTokens, avgEstPct, ranker.cfg.BatchTokens)
		}

		if valid {
			break
		}
		ranker.cfg.BatchSize--
		log.Printf("Decreasing batch size to %d", ranker.cfg.BatchSize)
		if ranker.cfg.BatchSize == 0 {
			log.Fatal("Cannot create a valid batch within the token limit")
		}
	}
}

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
		length = len(base64Encoded)
	}
	return base64Encoded[:length]
}

func loadObjectsFromFile(filePath string, templateData string) (objects []Object, err error) {
	var tmpl *template.Template
	if templateData != "" {
		if templateData[0] == '@' {
			content, err := os.ReadFile(templateData[1:])
			if err != nil {
				return nil, err
			}
			templateData = string(content)
		}
		if tmpl, err = template.New("raink-item-template").Parse(templateData); err != nil {
			return nil, err
		}
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(filePath))
	if ext == ".json" {
		// parse the file in an opaque array
		var data []interface{}
		if err := json.NewDecoder(file).Decode(&data); err != nil {
			return nil, err
		}

		// iterate over the map and create objects
		for _, value := range data {
			var valueStr string
			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, value); err != nil {
					return nil, err
				}
				valueStr = tmplData.String()
			} else {
				log.Printf("WARNING: using json input without a template, using JSON object as it is\n")
				jsonValue, err := json.Marshal(value)
				if err != nil {
					return nil, err
				}
				valueStr = string(jsonValue)
			}

			id := ShortDeterministicID(valueStr, idLen)
			objects = append(objects, Object{ID: id, Value: valueStr})
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
				return nil, err
			}
			line = strings.TrimSpace(line)

			if tmpl != nil {
				var tmplData bytes.Buffer
				if err := tmpl.Execute(&tmplData, map[string]string{"Data": line}); err != nil {
					return nil, err
				}
				line = tmplData.String()
			}

			id := ShortDeterministicID(line, idLen)
			objects = append(objects, Object{ID: id, Value: line})
		}
	}

	return objects, nil
}

// TODO: Move all of this CLI-related code to a separate package.
func main() {
	log.SetOutput(os.Stderr)

	inputFile := flag.String("f", "", "Input file")
	inputTemplate := flag.String("template", "{{.Data}}", "Template for each object in the input file (prefix with @ to use a file)")
	batchSize := flag.Int("s", 10, "Number of items per batch")
	numRuns := flag.Int("r", 10, "Number of runs")
	batchTokens := flag.Int("t", 128000, "Max tokens per batch")
	initialPrompt := flag.String("p", "", "Initial prompt (prefix with @ to use a file)")
	outputFile := flag.String("o", "", "JSON output file")

	ollamaURL := flag.String("ollama-url", "http://localhost:11434/api/chat", "Ollama API URL")
	ollamaModel := flag.String("ollama-model", "", "Ollama model name (if not set, OpenAI will be used)")
	oaiModel := flag.String("openai-model", openai.ChatModelGPT4oMini, "OpenAI model name")
	encoding := flag.String("encoding", "o200k_base", "Tokenizer encoding")

	dryRun := flag.Bool("dry-run", false, "Enable dry run mode (log API calls without making them)")
	refinementRatio := flag.Float64("ratio", 0.5, "Refinement ratio as a decimal (e.g., 0.5 for 50%)")
	flag.Parse()

	// TODO: This should be a more resilient check. We're assuming that if the
	// batchTokens is 128000, then a user didn't pass that value via CLI (i.e.,
	// that it's the default value).
	if *ollamaModel != "" && *batchTokens == 128000 {
		*batchTokens = 4096
	}

	// This "threshold" is a way to add some padding to our estimation of
	// average token usage per batch. We're effectively leaving 5% of
	// wiggle room.
	var tokenLimitThreshold = int(0.95 * float64(*batchTokens))

	if *inputFile == "" {
		log.Println("Usage: raink -f <input_file> [-s <batch_size>] [-r <num_runs>] [-p <initial_prompt>] [-t <batch_tokens>] [-ollama-model <model_name>] [-ratio <refinement_ratio>]")
		return
	}

	if *refinementRatio < 0 || *refinementRatio >= 1 {
		fmt.Println("Error: Refinement ratio must be >= 0 and < 1")
		os.Exit(1)
	}

	userPrompt := *initialPrompt
	if strings.HasPrefix(userPrompt, "@") {
		filePath := strings.TrimPrefix(userPrompt, "@")
		content, err := os.ReadFile(filePath)
		if err != nil {
			log.Fatalf("Error reading initial prompt file: %v", err)
		}
		userPrompt = string(content)
	}

	config := &Config{
		InitialPrompt:   userPrompt,
		BatchSize:       *batchSize,
		NumRuns:         *numRuns,
		OllamaModel:     *ollamaModel,
		OpenAIModel:     *oaiModel,
		TokenLimit:      tokenLimitThreshold,
		RefinementRatio: *refinementRatio,
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		OllamaAPIURL:    *ollamaURL,
		Encoding:        *encoding,
		BatchTokens:     *batchTokens,
		DryRun:          *dryRun,
	}

	ranker, err := NewRanker(config)
	if err != nil {
		log.Fatal(err)
	}

	objects, err := loadObjectsFromFile(*inputFile, *inputTemplate)
	if err != nil {
		log.Fatal(err)
	}

	// check that no object is too large
	for _, obj := range objects {
		tokens := ranker.estimateTokens([]Object{obj})
		if tokens > *batchTokens {
			log.Fatalf("Object is too large with %d tokens:\n%s", tokens, obj.Value)
		}
	}

	// Dynamically adjust batch size upfront.
	ranker.AdjustBatchSize(objects, 10)

	// Recursive processing
	finalResults := ranker.Rank(objects, 1)

	// Add the rank key to each final result based on its position in the list
	for i := range finalResults {
		finalResults[i].Rank = i + 1
	}

	jsonResults, err := json.MarshalIndent(finalResults, "", "  ")
	if err != nil {
		panic(err)
	}

	if !config.DryRun {
		fmt.Println(string(jsonResults))
	}

	if *outputFile != "" {
		os.WriteFile(*outputFile, jsonResults, 0644)
		log.Printf("Results written to %s\n", *outputFile)
	}
}

// TODO: The final exposure value should be the sum of all exposures from all
// refinement rounds (not just the last one). This isn't crucial since exposure
// is just a helpful metric to show that objects compared to a sufficiently
// large number of other objects.

func (r *Ranker) Rank(objects []Object, round int) []FinalResult {
	r.round = round

	log.Printf("Round %d: Ranking %d objects\n", r.round, len(objects))

	// If we've narrowed down to a single object, we're done.
	if len(objects) == 1 {
		return []FinalResult{
			{
				Key:      objects[0].ID,
				Value:    objects[0].Value,
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

	var topPortionObjects []Object
	for _, result := range topPortion {
		topPortionObjects = append(topPortionObjects, Object{ID: result.Key, Value: result.Value})
	}

	refinedTopPortion := r.Rank(topPortionObjects, round+1)

	// Adjust scores by recursion depth; this serves as an inverted weight so
	// that later rounds are guaranteed to sit higher in the final list.
	for i := range refinedTopPortion {
		refinedTopPortion[i].Score /= float64(2 * round)
	}

	// Combine the refined top portion with the unrefined bottom portion.
	finalResults := append(refinedTopPortion, bottomPortion...)

	return finalResults
}

// TODO: Also log the request/retry attempt number.
func (r *Ranker) logFromApiCall(runNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf("Round %d, Run %*d/%d, Batch %*d/%d: "+message, r.round, len(strconv.Itoa(r.cfg.NumRuns)), runNum, r.cfg.NumRuns, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches)
	log.Printf(formattedMessage, args...)
}

func (r *Ranker) shuffleBatchRank(objects []Object) []FinalResult {
	scores := make(map[string][]float64)

	exposureCounts := make(map[string]int)

	resultsChan := make(chan []RankedObject, r.numBatches)

	var firstRunRemainderItems []Object

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
			go func(runNumber, batchNumber int, batch []Object) {
				rankedBatch := r.rankObjects(batch, runNumber, batchNumber)
				resultsChan <- rankedBatch
			}(i+1, j+1, batch)
		}

		// Collect results from all batches
		for j := 0; j < r.numBatches; j++ {
			rankedBatch := <-resultsChan
			for _, rankedObject := range rankedBatch {
				scores[rankedObject.Object.ID] = append(scores[rankedObject.Object.ID], rankedObject.Score)
				exposureCounts[rankedObject.Object.ID]++ // Update exposure count
			}
		}

		// Save remainder items from the first run
		if i == 0 {
			remainderStart := r.numBatches * r.cfg.BatchSize
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

func (r *Ranker) logTokenSizes(group []Object) {
	log.Println("Logging token sizes for each object in the batch:")
	for _, obj := range group {
		tokenSize := r.estimateTokens([]Object{obj})
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

func (r *Ranker) estimateTokens(group []Object) int {
	prompt := r.cfg.InitialPrompt + promptDisclaimer
	for _, obj := range group {
		prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}

	if r.cfg.OllamaModel != "" {
		// TODO: Update to use Ollama tokenize API when this PR is merged:
		// https://github.com/ollama/ollama/pull/6586
		return len(prompt) / 4
	} else {
		return len(r.encoding.Encode(prompt, nil, nil))
	}
}

func (r *Ranker) rankObjects(group []Object, runNumber int, batchNumber int) []RankedObject {
	prompt := r.cfg.InitialPrompt + promptDisclaimer
	for _, obj := range group {
		prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
	}

	if r.cfg.DryRun {
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
	if r.cfg.OllamaModel != "" {
		rankedResponse = r.callOllama(prompt, runNumber, batchNumber, inputIDs)
	} else {
		rankedResponse = r.callOpenAI(prompt, runNumber, batchNumber, inputIDs)
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

// Updates the rankedResponse in place to fix case-insensitive ID mismatches.
// If any IDs are missing, returns the missing IDs along with an error.
// TODO: Also error on IDs in rankedResponse that are not in inputIDs. For example:
// Run  1/10, Batch  8/10: Missing IDs: [VkCMOyV9]
// Ollama API response: {"objects": ["5reULTRv", "KTJsPKHz", "eBFIaWo7", "AhqhnGsE", "Ug_hOxYp", "bWfMDUnE", "4sSg4Ojz", "VkJMOyV9", "UJ1-iMmW", "v6Puwf8K"]}

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

func (r *Ranker) callOpenAI(prompt string, runNum int, batchNum int, inputIDs map[string]bool) RankedObjectResponse {

	customTransport := &CustomTransport{Transport: http.DefaultTransport}
	customClient := &http.Client{Transport: customTransport}

	client := openai.NewClient(
		option.WithAPIKey(r.cfg.OpenAIKey),
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

			return rankedResponse
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
			log.Fatalf("Run %*d/%d, Batch %*d/%d: Unexpected error: %v", len(strconv.Itoa(r.cfg.NumRuns)), runNum, r.cfg.NumRuns, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches, err)
		}
	}
}

func (r *Ranker) callOllama(prompt string, runNum int, batchNum int, inputIDs map[string]bool) RankedObjectResponse {

	var rankedResponse RankedObjectResponse

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
			log.Fatalf("Error creating Ollama API request body: %v", err)
		}

		req, err := http.NewRequest("POST", r.cfg.OllamaAPIURL, bytes.NewReader(requestBody))
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

		return rankedResponse
	}
}
