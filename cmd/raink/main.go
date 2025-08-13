package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/noperator/raink/pkg/raink"
	"github.com/openai/openai-go"
)

func main() {
	inputFile := flag.String("f", "", "Input file")
	forceJSON := flag.Bool("json", false, "Force JSON parsing regardless of file extension")
	inputTemplate := flag.String("template", "{{.Data}}", "Template for each object in the input file (prefix with @ to use a file)")
	batchSize := flag.Int("s", 10, "Number of items per batch")
	numRuns := flag.Int("r", 10, "Number of runs")
	batchTokens := flag.Int("t", 128000, "Max tokens per batch")
	initialPrompt := flag.String("p", "", "Initial prompt (prefix with @ to use a file)")
	outputFile := flag.String("o", "", "JSON output file")

	ollamaURL := flag.String("ollama-url", "http://localhost:11434/api/chat", "Ollama API URL")
	ollamaModel := flag.String("ollama-model", "", "Ollama model name (if not set, OpenAI will be used)")
	oaiModel := flag.String("openai-model", openai.ChatModelGPT4oMini, "OpenAI model name")
	oaiURL := flag.String("openai-url", "", "OpenAI API base URL (e.g., for OpenAI-compatible API like vLLM)")
	encoding := flag.String("encoding", "o200k_base", "Tokenizer encoding")

	dryRun := flag.Bool("dry-run", false, "Enable dry run mode (log API calls without making them)")
	refinementRatio := flag.Float64("ratio", 0.5, "Refinement ratio as a decimal (e.g., 0.5 for 50%)")
	debug := flag.Bool("debug", false, "Enable debug logging")
	flag.Parse()

	// Set up structured logging with level based on debug flag
	logLevel := slog.LevelInfo
	if *debug {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})).With("component", "raink-cli")

	// This is a heuristic to detect if the user set a custom batch token limit for Ollama
	if *ollamaModel != "" && *batchTokens == 128000 {
		*batchTokens = 4096
	}

	// This "threshold" is a way to add some padding to our estimation of
	// average token usage per batch. We're effectively leaving 5% of
	// wiggle room.
	var tokenLimitThreshold = int(0.95 * float64(*batchTokens))

	if *inputFile == "" {
		logger.Error("Usage: raink -f <input_file> [-s <batch_size>] [-r <num_runs>] [-p <initial_prompt>] [-t <batch_tokens>] [-ollama-model <model_name>] [-openai-model <model_name>] [-openai-url <base_url>] [-ratio <refinement_ratio>]")
		return
	}

	if *refinementRatio < 0 || *refinementRatio >= 1 {
		fmt.Println("refinement ratio must be >= 0 and < 1")
		os.Exit(1)
	}

	userPrompt := *initialPrompt
	if strings.HasPrefix(userPrompt, "@") {
		filePath := strings.TrimPrefix(userPrompt, "@")
		content, err := os.ReadFile(filePath)
		if err != nil {
			logger.Error("could not read initial prompt file", "error", err)
			os.Exit(1)
		}
		userPrompt = string(content)
	}

	config := &raink.Config{
		InitialPrompt:   userPrompt,
		BatchSize:       *batchSize,
		NumRuns:         *numRuns,
		OllamaModel:     *ollamaModel,
		OpenAIModel:     *oaiModel,
		TokenLimit:      tokenLimitThreshold,
		RefinementRatio: *refinementRatio,
		OpenAIKey:       os.Getenv("OPENAI_API_KEY"),
		OpenAIAPIURL:    *oaiURL,
		OllamaAPIURL:    *ollamaURL,
		Encoding:        *encoding,
		BatchTokens:     *batchTokens,
		DryRun:          *dryRun,
		LogLevel:        logLevel,
	}

	ranker, err := raink.NewRanker(config)
	if err != nil {
		logger.Error("failed to create ranker", "error", err)
		os.Exit(1)
	}

	finalResults, err := ranker.RankFromFile(*inputFile, *inputTemplate, *forceJSON)
	if err != nil {
		logger.Error("failed to rank from file", "error", err)
		os.Exit(1)
	}

	jsonResults, err := json.MarshalIndent(finalResults, "", "  ")
	if err != nil {
		logger.Error("could not marshal results to JSON", "error", err)
		os.Exit(1)
	}

	if !config.DryRun {
		fmt.Println(string(jsonResults))
	}

	if *outputFile != "" {
		os.WriteFile(*outputFile, jsonResults, 0644)
		logger.Info("results written to file", "file", *outputFile)
	}
}
