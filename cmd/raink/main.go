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

// Helper function to use fallback values for two-tier parameters
func valueOrDefaultFloat(value, defaultValue float64) float64 {
	if value == 0 {
		return defaultValue
	}
	return value
}

func valueOrDefaultInt(value, defaultValue int) int {
	if value == 0 {
		return defaultValue
	}
	return value
}

func main() {
	inputFile := flag.String("f", "", "Input file")
	forceJSON := flag.Bool("json", false, "Force JSON parsing regardless of file extension")
	inputTemplate := flag.String("template", "{{.Data}}", "Template for each object in the input file (prefix with @ to use a file)")
	batchSize := flag.Int("s", 10, "Number of items per batch")
	batchTokens := flag.Int("t", 128000, "Max tokens per batch")
	initialPrompt := flag.String("p", "", "Initial prompt (prefix with @ to use a file)")
	outputFile := flag.String("o", "", "JSON output file")

	oaiModel := flag.String("openai-model", openai.ChatModelGPT4oMini, "OpenAI model name")
	oaiURL := flag.String("openai-url", "", "OpenAI API base URL (e.g., for OpenAI-compatible API like vLLM)")
	encoding := flag.String("encoding", "o200k_base", "Tokenizer encoding")

	dryRun := flag.Bool("dry-run", false, "Enable dry run mode (log API calls without making them)")
	observe := flag.Bool("observe", false, "Enable live visualization of ranking process")
	roundPause := flag.Int("round-pause", 0, "Seconds to pause after each round completes (useful for analysis/screenshots)")
	runPause := flag.Int("run-pause", 0, "Seconds to pause after each iteration/run (useful for analysis/screenshots)")
	stdDevItem := flag.Float64("stddev-item", 1.41, "Standard deviation threshold for item convergence (default: 1.41)")
	cutoffRange := flag.Float64("cutoff-range", 0.05, "Maximum cutoff range as percentage of dataset size (e.g., 0.05 = 5%) (default: 0.05)")
	cutoffWindow := flag.Int("cutoff-window", 7, "Sliding window size for tracking recent elbow positions (default: 7)")
	blockTolerance := flag.Int("block-tolerance", 0, "Number of non-converged items that can be skipped when building top/bottom blocks (default: 0 for strict contiguous)")
	minRuns := flag.Int("min-runs", 0, "Minimum number of iterations per round (default: 0 = log2(n))")
	maxRuns := flag.Int("max-runs", 0, "Maximum number of iterations per round (default: 0 = sqrt(n))")
	stableRuns := flag.Int("stable-runs", 3, "Number of consecutive stable iterations required for convergence (default: 3)")
	batchConcurrency := flag.Int("batch-concurrency", 5, "Number of batches to process concurrently (default: 5)")

	// Threshing parameters (Round 1)
	threshStdDev := flag.Float64("thresh-stddev", 0, "Standard deviation threshold for threshing round (0 = use stddev-item)")
	threshCutoffWindow := flag.Int("thresh-cutoff-window", 0, "Sliding window size for threshing (0 = use cutoff-window)")
	threshCutoffRange := flag.Float64("thresh-cutoff-range", 0, "Cutoff range percentage for threshing (0 = use cutoff-range)")
	threshStableRuns := flag.Int("thresh-stable-runs", 0, "Stable runs required for threshing (0 = use stable-runs)")

	// Ranking parameters (Round 2+)
	rankStdDev := flag.Float64("rank-stddev", 0, "Standard deviation threshold for ranking rounds (0 = use stddev-item)")
	rankCutoffWindow := flag.Int("rank-cutoff-window", 0, "Sliding window size for ranking (0 = use cutoff-window)")
	rankCutoffRange := flag.Float64("rank-cutoff-range", 0, "Cutoff range percentage for ranking (0 = use cutoff-range)")
	rankStableRuns := flag.Int("rank-stable-runs", 0, "Stable runs required for ranking (0 = use stable-runs)")

	// Ranking-specific cutoff parameters
	rankCutoffRatio := flag.Float64("rank-cutoff-ratio", 0.5, "Fixed cutoff ratio for ranking rounds (e.g., 0.5 for top 50%)")
	rankPlateauThreshold := flag.Float64("rank-plateau-threshold", 0.02, "Rate of change threshold for plateau detection (e.g., 0.02 for 2%)")
	rankElbowThreshold := flag.Int("rank-elbow-threshold", 25, "Dataset size threshold for switching from elbow to ratio cutoff in ranking rounds (default: 25)")

	debug := flag.Bool("debug", false, "Enable debug logging")
	invertCutoff := flag.Bool("invert-cutoff", false, "Invert cutoff selection to refine bottom portion instead of top (experimental)")
	stddevElbow := flag.Bool("stddev-elbow", false, "Use standard deviation elbow for cutoff instead of rank elbow (experimental)")
	stopOnNoSignal := flag.Bool("stop-no-signal", true, "Stop recursion when no stable items found after cutoff convergence (default: true)")
	noSignalThreshold := flag.Float64("no-signal-threshold", 0.0, "Minimum percentage of stable items required to continue (0.0 = any stable item)")
	progressiveViz := flag.Bool("progressive", false, "Use progressive visualization (keeps previous rounds visible)")
	minimap := flag.Bool("minimap", false, "Show minimap panel showing current round data (only with -observe)")
	minimapAll := flag.Bool("minimap-all", false, "Show minimap with complete dataset including previous rounds (only with -observe)")
	reasoning := flag.Bool("reasoning", false, "Collect and summarize reasoning for rankings (skips round 1)")
	flag.Parse()

	// Set up structured logging with level based on debug flag
	logLevel := slog.LevelInfo
	if *debug {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: logLevel,
	})).With("component", "raink-cli")

	// This "threshold" is a way to add some padding to our estimation of
	// average token usage per batch. We're effectively leaving 5% of
	// wiggle room.
	var tokenLimitThreshold = int(0.95 * float64(*batchTokens))

	if *inputFile == "" {
		logger.Error("Usage: raink -f <input_file> [-s <batch_size>] [-p <initial_prompt>] [-t <batch_tokens>] [-openai-model <model_name>] [-openai-url <base_url>] [-observe] [-round-pause <seconds>] [-run-pause <seconds>] [-stddev-item <float>] [-cutoff-range <float>] [-cutoff-window <int>] [-stable-runs <int>] [-batch-concurrency <int>] [-block-tolerance <int>] [-min-runs <int>] [-max-runs <int>]")
		return
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
		InitialPrompt:     userPrompt,
		BatchSize:         *batchSize,
		OpenAIModel:       *oaiModel,
		TokenLimit:        tokenLimitThreshold,
		OpenAIKey:         os.Getenv("OPENAI_API_KEY"),
		OpenAIAPIURL:      *oaiURL,
		Encoding:          *encoding,
		BatchTokens:       *batchTokens,
		DryRun:            *dryRun,
		LogLevel:        logLevel,
		Observe:         *observe,
		RoundPause:      *roundPause,
		RunPause:        *runPause,
		StdDevItem:         *stdDevItem,
		CutoffRangePercent: *cutoffRange,
		CutoffWindow:       *cutoffWindow,
		BlockTolerance:  *blockTolerance,
		MinRuns:         *minRuns,
		MaxRuns:         *maxRuns,
		StableRuns:      *stableRuns,
		BatchConcurrency: *batchConcurrency,
		InvertCutoff:      *invertCutoff,
		StdDevElbow:       *stddevElbow,
		StopOnNoSignal:    *stopOnNoSignal,
		NoSignalThreshold: *noSignalThreshold,
		ProgressiveViz:    *progressiveViz,
		MinimapEnabled:    *minimap || *minimapAll,
		MinimapShowAll:    *minimapAll,

		// Threshing parameters - fallback to general params if not specified
		ThreshStdDev:       valueOrDefaultFloat(*threshStdDev, *stdDevItem),
		ThreshCutoffWindow: valueOrDefaultInt(*threshCutoffWindow, *cutoffWindow),
		ThreshCutoffRange:  valueOrDefaultFloat(*threshCutoffRange, *cutoffRange),
		ThreshStableRuns:   valueOrDefaultInt(*threshStableRuns, *stableRuns),

		// Ranking parameters - fallback to general params if not specified
		RankStdDev:       valueOrDefaultFloat(*rankStdDev, *stdDevItem),
		RankCutoffWindow: valueOrDefaultInt(*rankCutoffWindow, *cutoffWindow),
		RankCutoffRange:  valueOrDefaultFloat(*rankCutoffRange, *cutoffRange),
		RankStableRuns:   valueOrDefaultInt(*rankStableRuns, *stableRuns),

		// Ranking-specific cutoff parameters
		RankCutoffRatio:      *rankCutoffRatio,
		RankPlateauThreshold: *rankPlateauThreshold,
		RankElbowThreshold:   *rankElbowThreshold,

		// Reasoning
		Reasoning: *reasoning,
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

	// Only print to stdout if not in observe mode (terminal state interferes)
	if !config.Observe {
		fmt.Println(string(jsonResults))
	}

	if *outputFile != "" {
		os.WriteFile(*outputFile, jsonResults, 0644)
		logger.Info("results written to file", "file", *outputFile)
	} else if config.Observe {
		logger.Info("Observe mode: use -o flag to write JSON results to file")
	}
}
