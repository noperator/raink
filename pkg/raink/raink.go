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
	"log/slog"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/pkoukk/tiktoken-go"
)

const (
	idLen        = 8
	minBatchSize = 2
)

// Word lists for generating memorable IDs
var (
	adjectives = []string{
		"apt", "bad", "big", "coy", "dim", "dry", "far", "fat", "fit", "fun",
		"hip", "hot", "icy", "lax", "low", "mad", "mid", "net", "new", "old",
		"pat", "raw", "red", "sad", "shy", "tan", "wet",
	}
	nouns = []string{
		"act", "age", "aid", "air", "ant", "ape", "arm", "art", "ash", "bag",
		"bar", "bat", "bay", "bed", "bet", "bid", "bin", "bit", "bog", "bow",
		"box", "boy", "bud", "bug", "bun", "bus", "can", "cap", "car", "cat",
		"cob", "cot", "cow", "cub", "cup", "cut", "dad", "dam", "day", "den",
		"dew", "dog", "dot", "ear", "elf", "elk", "elm", "emu", "end", "era",
		"eye", "fan", "fax", "fig", "fix", "flu", "fly", "fob", "fog", "fox",
		"fur", "gap", "gas", "gem", "gum", "guy", "gym", "hat", "hay", "hen",
		"hip", "hit", "hog", "hot", "hut", "ice", "ink", "jam", "jar", "jaw",
		"job", "joy", "jug", "keg", "key", "kid", "lab", "lap", "law", "leg",
		"lie", "lip", "log", "lot", "man", "map", "mat", "mix", "mom", "mop",
		"mud", "mug", "net", "nut", "oak", "oar", "oil", "one", "owl", "pad",
		"pan", "paw", "pea", "pen", "pet", "pew", "pie", "pig", "pin", "pop",
		"pot", "rag", "ram", "rat", "ray", "rim", "rip", "rod", "row", "rub",
		"rug", "rum", "run", "saw", "sea", "sir", "sky", "son", "sow", "soy",
		"spy", "sun", "tax", "tea", "tie", "tin", "tip", "toe", "tom", "ton",
		"top", "toy", "tub", "urn", "van", "wad", "war", "wax", "way", "web",
		"yak", "yam",
	}
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
	Logger          *slog.Logger     `json:"-"`
	LogLevel        slog.Level       `json:"-"` // Defaults to 0 (slog.LevelInfo)
}

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

func NewRanker(config *Config) (*Ranker, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	// Initialize default logger if not provided
	if config.Logger == nil {
		config.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
			Level:     config.LogLevel,
			AddSource: false,
		})).With("component", "raink")
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

// dynamically adjust batch size to fit within token limits
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
					ranker.cfg.Logger.Debug("Sample exceeded token threshold - estimated tokens > max limit", "sample", i, "estimated_tokens", estBatchTokens, "max_threshold", ranker.cfg.TokenLimit)
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
			ranker.cfg.Logger.Debug("Average estimated tokens calculated", "tokens", avgEstTokens, "percentage_of_max", avgEstPct, "max_tokens", ranker.cfg.TokenLimit)
			break
		}
		if ranker.cfg.BatchSize <= minBatchSize {
			return fmt.Errorf("cannot create a valid batch within the token limit")
		}
		ranker.cfg.BatchSize--
		ranker.cfg.Logger.Debug("Decreasing batch size to fit within token limits", "new_size", ranker.cfg.BatchSize)
	}
	return nil
}

type object struct {
	ID     string      `json:"id"`
	Value  string      `json:"value"`  // to be ranked
	Object interface{} `json:"object"` // if loading from json file
}

type rankedObject struct {
	Object object
	Score  float64
}

type rankedObjectResponse struct {
	Objects []string `json:"objects" jsonschema_description:"List of ranked object IDs"`
}

type RankedObject struct {
	Key      string      `json:"key"`
	Value    string      `json:"value"`
	Object   interface{} `json:"object"` // if loading from json file
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

// createIDMappings generates memorable temporary IDs for a batch of objects
func createIDMappings(objects []object, rng *rand.Rand, logger *slog.Logger) (map[string]string, map[string]string, error) {
	originalToTemp := make(map[string]string)
	tempToOriginal := make(map[string]string)
	usedCombos := make(map[string]bool)

	maxAttempts := len(adjectives) * len(nouns) * 2 // Allow some randomness

	for _, obj := range objects {
		attempts := 0
		found := false

		for attempts < maxAttempts && !found {
			adj := adjectives[rng.Intn(len(adjectives))]
			noun := nouns[rng.Intn(len(nouns))]
			combination := adj + noun

			// Check for consecutively repeated characters
			hasRepeats := false
			for i := 0; i < len(combination)-1; i++ {
				if combination[i] == combination[i+1] {
					hasRepeats = true
					break
				}
			}

			// If no repeats and not used, use this combination
			if !hasRepeats && !usedCombos[combination] {
				usedCombos[combination] = true
				originalToTemp[obj.ID] = combination
				tempToOriginal[combination] = obj.ID
				found = true
			}

			attempts++
		}

		if !found {
			// Fall back to original IDs if we can't generate memorable ones
			logger.Warn("Failed to generate memorable IDs, falling back to original IDs", "error", "unable to generate unique memorable ID")
			return nil, nil, fmt.Errorf("unable to generate unique memorable ID after %d attempts", maxAttempts)
		}
	}

	return originalToTemp, tempToOriginal, nil
}

// translateIDsInResponse translates temporary IDs back to original IDs in the response
func translateIDsInResponse(response *rankedObjectResponse, tempToOriginal map[string]string) {
	for i, id := range response.Objects {
		if originalID, exists := tempToOriginal[id]; exists {
			response.Objects[i] = originalID
		}
	}
}

var rankedObjectResponseSchema = generateSchema[rankedObjectResponse]()

// ShortDeterministicID generates a deterministic ID of specified length from input string.
// It uses SHA-256 hash and Base64 encoding, keeping only alphanumeric characters.
func ShortDeterministicID(input string, length int) string {
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

// ranks objects loaded from a file with optional template
func (r *Ranker) RankFromFile(filePath string, templateData string, forceJSON bool) ([]RankedObject, error) {
	objects, err := r.loadObjectsFromFile(filePath, templateData, forceJSON)
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

func (r *Ranker) loadObjectsFromFile(filePath string, templateData string, forceJSON bool) (objects []object, err error) {
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
				r.cfg.Logger.Warn("using json input without a template, using JSON object as it is")
				jsonValue, err := json.Marshal(value)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal JSON value: %w", err)
				}
				valueStr = string(jsonValue)
			}

			id := ShortDeterministicID(valueStr, idLen)
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

			id := ShortDeterministicID(line, idLen)
			objects = append(objects, object{ID: id, Object: nil, Value: line})
		}
	}

	return objects, nil
}

// perform the ranking algorithm on the given objects
func (r *Ranker) rank(objects []object, round int) []RankedObject {
	r.round = round

	r.cfg.Logger.Info("Ranking objects", "round", r.round, "count", len(objects))

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

	// Ensure we have at least 2 objects for meaningful ranking
	// (you need at least 2 items to rank against each other)
	if mid < 2 {
		return results
	}

	topPortion := results[:mid]
	bottomPortion := results[mid:]

	// If we haven't reduced the number of objects (as may eventually happen
	// for a ratio above 0.5), we're done.
	if len(topPortion) == len(objects) {
		return results
	}

	r.cfg.Logger.Debug("Top items being sent back into recursion:")
	for i, obj := range topPortion {
		r.cfg.Logger.Debug("Recursive item", "rank", i+1, "id", obj.Key, "score", obj.Score, "value", obj.Value)
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
	formattedMessage := fmt.Sprintf(message, args...)
	r.cfg.Logger.Debug(formattedMessage, "round", r.round, "run", runNum, "total_runs", r.cfg.NumRuns, "batch", batchNum, "total_batches", r.numBatches)
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
							r.cfg.Logger.Debug("Conflicting remainder item found", "current", item, "first_run", firstRunItem)
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
		r.cfg.Logger.Debug("Submitting batches to API", "round", r.round, "run", i+1, "total_runs", r.cfg.NumRuns)
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
				r.cfg.Logger.Error("Error in batch processing", "error", result.err)
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
				r.cfg.Logger.Debug("First run remainder items", "items", firstRunRemainderItems)
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
	r.cfg.Logger.Debug("Logging token sizes for each object in the batch:")
	for _, obj := range group {
		tokenSize := r.estimateTokens([]object{obj}, false)
		valuePreview := obj.Value
		if len(valuePreview) > 100 {
			valuePreview = valuePreview[:100]
		}
		r.cfg.Logger.Debug("Object token size", "id", obj.ID, "token_size", tokenSize, "value_preview", valuePreview)
	}
}

const objectFmt = `
<object>
<id>
%s
</id>
<value>
%s
</value>
</object>
`

const reminders = `- ALWAYS respond with the SHORT ID associated with each object!
— NEVER respond with the actual value!
- ALWAYS respond in JSON format as specified! {"objects": ["<ID1>", "<ID2>", ...]}
- Respond in RANKED DESCENDING order, where the FIRST item in your response is the MOST RELEVANT!
- ALWAYS return ALL of the IDs in the list!
- NEVER include backticks around IDs in your response!
— NEVER include scores or a written reason/justification in your response!`

const rankPrompt = `%s

Your job here is to RANK the incoming objects in DESCENDING order, where the FIRST item in your response is the MOST RELEVANT.
Rather than returning the full contents of each object, you'll simply return the SHORT ID (usually 6-8 chars) associated with each object (think of it like a simple "pointer" to the object).
You'll return these identifiers in the following JSON schema: {"objects": ["<ID1>", "<ID2>", ...]}

Here are the objects to be ranked:
<objects>
%s
</objects>

Let me remind you to be ABSOLUTELY SURE to:
%s`

const missingIDsStr = `Your last response was invalid for these reasons:
- You did not provide the following required IDs: [%s]
- You mistakenly provided the following invalid IDs: [%s]. (If this is empty, then just focus on the missing IDs.)

Try again, and make ABSOLUTELY SURE to remember to:
%s`

const invalidJSONStr = "Your last response was not valid JSON. Try again!"

func (r *Ranker) estimateTokens(group []object, includePrompt bool) int {
	initPrompt := ""
	if includePrompt {
		initPrompt = r.cfg.InitialPrompt
	}
	objectData := ""
	for _, obj := range group {
		objectData += fmt.Sprintf(objectFmt, obj.ID, obj.Value)
	}

	text := fmt.Sprintf(rankPrompt, initPrompt, objectData, reminders)

	if r.cfg.OllamaModel != "" {
		// TODO: Update to use Ollama tokenize API when this PR is merged:
		// https://github.com/ollama/ollama/pull/6586
		return len(text) / 4
	} else {
		return len(r.encoding.Encode(text, nil, nil))
	}
}

func (r *Ranker) rankObjects(group []object, runNumber int, batchNumber int) ([]rankedObject, error) {
	if r.cfg.DryRun {
		r.cfg.Logger.Debug("Dry run API call")
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

	maxRetries := 10
	for attempt := 0; attempt < maxRetries; attempt++ {
		// Try to create memorable ID mappings for each attempt
		originalToTemp, tempToOriginal, err := createIDMappings(group, r.rng, r.cfg.Logger)
		useMemorableIDs := err == nil && originalToTemp != nil && tempToOriginal != nil

		inputIDs := make(map[string]bool)

		objectData := ""
		if useMemorableIDs {
			// Use memorable IDs in the prompt
			for _, obj := range group {
				tempID := originalToTemp[obj.ID]
				objectData += fmt.Sprintf(objectFmt, tempID, obj.Value)
				inputIDs[tempID] = true
			}
		} else {
			// Fall back to original IDs
			for _, obj := range group {
				objectData += fmt.Sprintf(objectFmt, obj.ID, obj.Value)
				inputIDs[obj.ID] = true
			}
		}

		prompt := fmt.Sprintf(rankPrompt, r.cfg.InitialPrompt, objectData, reminders)

		var rankedResponse rankedObjectResponse
		if r.cfg.OllamaModel != "" {
			rankedResponse, err = r.callOllama(prompt, runNumber, batchNumber, inputIDs)
		} else {
			rankedResponse, err = r.callOpenAI(prompt, runNumber, batchNumber, inputIDs)
		}
		if err != nil {
			if attempt == maxRetries-1 {
				return nil, err
			}
			r.logFromApiCall(runNumber, batchNumber, "API call failed, retrying with new memorable IDs (attempt %d): %v", attempt+1, err)
			continue
		}

		// Translate temporary IDs back to original IDs if using memorable IDs
		if useMemorableIDs {
			translateIDsInResponse(&rankedResponse, tempToOriginal)
		}

		// Check if we got all expected IDs
		expectedIDs := make(map[string]bool)
		for _, obj := range group {
			expectedIDs[obj.ID] = true
		}
		for _, id := range rankedResponse.Objects {
			delete(expectedIDs, id)
		}

		if len(expectedIDs) > 0 {
			var missingIDs []string
			for id := range expectedIDs {
				missingIDs = append(missingIDs, id)
			}
			if attempt == maxRetries-1 {
				return nil, fmt.Errorf("missing IDs after %d attempts: %v", maxRetries, missingIDs)
			}
			r.logFromApiCall(runNumber, batchNumber, "Missing IDs, retrying with new memorable IDs (attempt %d): %v", attempt+1, missingIDs)
			continue
		}

		// Success! Assign scores based on position in the ranked list
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

	return nil, fmt.Errorf("failed after %d attempts", maxRetries)
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
func validateIDs(rankedResponse *rankedObjectResponse, inputIDs map[string]bool) ([]string, []string, error) {
	// Create a map for case-insensitive ID matching
	inputIDsLower := make(map[string]string)
	for id := range inputIDs {
		inputIDsLower[strings.ToLower(id)] = id
	}

	// IDs that were in the original provided list, but not returned by the LLM
	// Start by assuming that all IDs are missing, then prove wrong
	missingIDs := make(map[string]bool)
	for id := range inputIDs {
		missingIDs[id] = true
	}

	// IDs returned by the LLM that weren't in the original provided list
	wrongIDs := []string{}

	for i, id := range rankedResponse.Objects {
		nonAlpha := regexp.MustCompile(`[^a-zA-Z]+`)
		id := nonAlpha.ReplaceAllString(id, "")
		lowerID := strings.ToLower(id)

		// Check if this ID returned by the LLM was in the original provided list
		if correctID, found := inputIDsLower[lowerID]; found {
			if correctID != id {
				// Replace the case-wrong match with the correct ID
				rankedResponse.Objects[i] = correctID
			}
			delete(missingIDs, correctID)
		} else {
			wrongIDs = append(wrongIDs, id)
		}
	}

	// If we got back all the IDs we needed, then it doesn't matter if it returned extra "wrong" IDs
	if len(missingIDs) == 0 {
		return nil, nil, nil
	} else {
		missingIDsKeys := make([]string, 0, len(missingIDs))
		for id := range missingIDs {
			missingIDsKeys = append(missingIDsKeys, id)
		}
		return missingIDsKeys, wrongIDs, fmt.Errorf("missing IDs: %s; wrong IDs: %s", strings.Join(missingIDsKeys, ", "), strings.Join(wrongIDs, ", "))
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
	maxRetries := 3
	attempts := 0
	for attempts < maxRetries {
		attempts++
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Messages: conversationHistory,
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "ranked_object_response",
						Description: openai.String("List of ranked object IDs"),
						Schema:      rankedObjectResponseSchema,
						Strict:      openai.Bool(true),
					},
				},
			},
			Model: r.cfg.OpenAIModel,
		})
		if err == nil {

			conversationHistory = append(conversationHistory,
				openai.AssistantMessage(completion.Choices[0].Message.Content),
			)

			err = json.Unmarshal([]byte(completion.Choices[0].Message.Content), &rankedResponse)
			if err != nil {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Error unmarshalling response: %v\n", err))
				if attempts >= maxRetries {
					return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts: invalid JSON response", maxRetries)
				}
				conversationHistory = append(conversationHistory,
					openai.UserMessage(invalidJSONStr),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				r.cfg.Logger.Debug("OpenAI API response", "content", trimmedContent)
				continue
			}

			missingIDs, wrongIDs, err := validateIDs(&rankedResponse, inputIDs)
			if err != nil {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Missing IDs: [%s] Wrong IDs: [%s]", strings.Join(missingIDs, ", "), strings.Join(wrongIDs, ", ")))
				if attempts >= maxRetries {
					return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts: missing %v, wrong %v", maxRetries, missingIDs, wrongIDs)
				}
				conversationHistory = append(conversationHistory,
					openai.UserMessage(fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "), strings.Join(wrongIDs, ", "), reminders)),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				r.cfg.Logger.Debug("OpenAI API response", "content", trimmedContent)
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
	return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts", maxRetries)
}

func (r *Ranker) callOllama(prompt string, runNum int, batchNum int, inputIDs map[string]bool) (rankedObjectResponse, error) {

	var rankedResponse rankedObjectResponse

	// Initialize the conversation history with the initial prompt
	conversationHistory := []map[string]interface{}{
		{"role": "user", "content": prompt},
	}

	maxRetries := 3
	attempts := 0
	for attempts < maxRetries {
		attempts++

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
			if attempts >= maxRetries {
				return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts: invalid JSON response", maxRetries)
			}
			conversationHistory = append(conversationHistory,
				map[string]interface{}{
					"role":    "user",
					"content": invalidJSONStr,
				},
			)
			trimmedContent := strings.TrimSpace(ollamaResponse.Message.Content)
			r.cfg.Logger.Debug("Ollama API response", "content", trimmedContent)
			continue
		}

		missingIDs, wrongIDs, err := validateIDs(&rankedResponse, inputIDs)
		if err != nil {
			r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Missing IDs: [%s] Wrong IDs: [%s]", strings.Join(missingIDs, ", "), strings.Join(wrongIDs, ", ")))
			if attempts >= maxRetries {
				return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts: missing %v, wrong %v", maxRetries, missingIDs, wrongIDs)
			}
			conversationHistory = append(conversationHistory,
				map[string]interface{}{
					"role":    "user",
					"content": fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "), strings.Join(wrongIDs, ", "), reminders),
				},
			)
			trimmedContent := strings.TrimSpace(ollamaResponse.Message.Content)
			r.cfg.Logger.Debug("Ollama API response", "content", trimmedContent)
			continue
		}

		return rankedResponse, nil
	}
	return rankedObjectResponse{}, fmt.Errorf("failed after %d conversation attempts", maxRetries)
}
