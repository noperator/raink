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
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"text/template"
	"time"

	"github.com/gdamore/tcell/v2"
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
	InitialPrompt     string           `json:"initial_prompt"`
	BatchSize         int              `json:"batch_size"`
	OpenAIModel       openai.ChatModel `json:"openai_model"`
	TokenLimit        int              `json:"token_limit"`
	OpenAIKey         string           `json:"-"`
	OpenAIAPIURL      string           `json:"-"`
	Encoding          string           `json:"encoding"`
	BatchTokens       int              `json:"batch_tokens"`
	DryRun            bool             `json:"-"`
	Logger            *slog.Logger     `json:"-"`
	LogLevel          slog.Level       `json:"-"` // Defaults to 0 (slog.LevelInfo)
	Observe         bool    `json:"observe"`
	RoundPause      int     `json:"round_pause"`      // Seconds to pause after each round
	RunPause        int     `json:"run_pause"`        // Seconds to pause after each iteration/run
	StdDevItem         float64 `json:"stddev_item"`         // Standard deviation threshold for item convergence
	CutoffRangePercent float64 `json:"cutoff_range_percent"` // Maximum cutoff range as percentage of dataset size
	CutoffWindow       int     `json:"cutoff_window"`       // Sliding window size for tracking recent elbow positions
	BlockTolerance  int     `json:"block_tolerance"`  // Number of items that can be skipped when building blocks
	MinRuns         int     `json:"min_runs"`         // Minimum iterations per round (0 = log2(n) default)
	MaxRuns         int     `json:"max_runs"`         // Maximum iterations per round (0 = sqrt(n) default)
	StableRuns      int     `json:"stable_runs"`      // Number of consecutive stable iterations required for convergence
	BatchConcurrency int    `json:"batch_concurrency"` // Number of batches to process concurrently
	InvertCutoff      bool    `json:"invert_cutoff"`      // Invert cutoff selection to refine bottom portion instead of top
	StdDevElbow       bool    `json:"stddev_elbow"`       // Use standard deviation elbow for cutoff instead of rank elbow
	StopOnNoSignal    bool    `json:"stop_on_no_signal"`  // Enable no-signal stopping condition
	NoSignalThreshold float64 `json:"no_signal_threshold"` // Minimum percentage of stable items required to continue
	ProgressiveViz    bool    `json:"progressive_viz"`     // Use progressive visualization mode
	MinimapEnabled    bool    `json:"minimap_enabled"`     // Show minimap panel
	MinimapShowAll    bool    `json:"minimap_show_all"`    // Show complete dataset in minimap (vs current round only)

	// Threshing parameters (Round 1)
	ThreshStdDev       float64 `json:"thresh_stddev"`        // Standard deviation threshold for threshing round
	ThreshCutoffWindow int     `json:"thresh_cutoff_window"` // Sliding window size for threshing
	ThreshCutoffRange  float64 `json:"thresh_cutoff_range"`  // Cutoff range percentage for threshing
	ThreshStableRuns   int     `json:"thresh_stable_runs"`   // Stable runs required for threshing

	// Ranking parameters (Round 2+)
	RankStdDev       float64 `json:"rank_stddev"`        // Standard deviation threshold for ranking rounds
	RankCutoffWindow int     `json:"rank_cutoff_window"` // Sliding window size for ranking
	RankCutoffRange  float64 `json:"rank_cutoff_range"`  // Cutoff range percentage for ranking
	RankStableRuns   int     `json:"rank_stable_runs"`   // Stable runs required for ranking

	// Ranking-specific cutoff parameters
	RankCutoffRatio      float64 `json:"rank_cutoff_ratio"`      // Fixed ratio for ranking rounds
	RankPlateauThreshold float64 `json:"rank_plateau_threshold"` // Rate of change threshold for convergence
	RankElbowThreshold   int     `json:"rank_elbow_threshold"`   // Dataset size threshold for switching to ratio cutoff

	// Reasoning
	Reasoning bool `json:"reasoning"` // Collect and summarize reasoning for rankings
}

// MinimapData represents the compressed dataset for minimap visualization
type MinimapData struct {
	TotalItems    int
	ElbowPosition int
	Buckets       []MinimapBucket
}

// MinimapBucket represents a compressed row in the minimap
type MinimapBucket struct {
	StartIdx       int
	EndIdx         int
	AvgRank        float64
	AvgStdDev      float64
	ItemCount      int
	ConvergedCount int
	ContainsElbow  bool
}

// StopReason represents why ranking stopped
type StopReason int

const (
	StopReasonNone StopReason = iota
	StopReasonSingleItem    // Only 1 item remains
	StopReasonNoMoreCuts    // len(topPortion) == len(objects)
	StopReasonTooFewItems   // len(topPortion) < 2
	StopReasonNoSignal      // No stable items after convergence
	StopReasonAllConverged  // All items converged (base case)
	StopReasonMaxDepth      // Hit maximum recursion depth (if implemented)
)

// String returns human-readable stop reason
func (s StopReason) String() string {
	switch s {
	case StopReasonSingleItem:
		return "Single item remaining - ranking complete"
	case StopReasonNoMoreCuts:
		return "No meaningful cutoff found - all items equivalent"
	case StopReasonTooFewItems:
		return "Too few items to continue refining"
	case StopReasonNoSignal:
		return "No stable items found - rankings are noise"
	case StopReasonAllConverged:
		return "All items fully converged"
	case StopReasonMaxDepth:
		return "Maximum recursion depth reached"
	default:
		return "Ranking completed"
	}
}

func (c *Config) Validate() error {
	if c.InitialPrompt == "" {
		return fmt.Errorf("initial prompt cannot be empty")
	}
	if c.BatchSize <= 0 {
		return fmt.Errorf("batch size must be greater than 0")
	}
	if c.TokenLimit <= 0 {
		return fmt.Errorf("token limit must be greater than 0")
	}
	// Only require API key if not using a custom endpoint
	if c.OpenAIAPIURL == "" && c.OpenAIKey == "" {
		return fmt.Errorf("openai key cannot be empty")
	}
	if c.BatchSize < minBatchSize {
		return fmt.Errorf("batch size must be at least %d", minBatchSize)
	}
	return nil
}

type Ranker struct {
	cfg                    *Config
	encoding               *tiktoken.Tiktoken
	rng                    *rand.Rand
	numBatches             int
	round                  int
	rounds                 []roundInfo
	screen                 tcell.Screen
	pendingRoundPause      *roundPauseInfo
	lastCutoffMethod       string
	recentElbowPositions   []int
	consecutiveStableCount int
	lastStableCutoff       int
	finalStopReason        StopReason
	savedAPICalls          int // Track API calls saved by early termination
	progressiveMode        bool
	allItemStats           map[string]*itemStats // Cumulative stats for all items across all rounds
	roundCutoffs          []RoundCutoff
	displayStates         map[string]*ItemDisplayState
	allRoundStats         map[string]*itemStats  // Cumulative stats across rounds

	// For plateau detection in ranking rounds
	previousAvgStdDev    float64
	plateauStableCount   int
}

type itemStats struct {
	ID                  string
	Value               string
	Object              interface{}
	rankHistory         []float64
	avgRank             float64
	stdDev              float64
	prevStdDev          float64
	noSignal            bool
	highConfidenceGroup bool
	reasoningSnippets   []string // Collected reasoning across all batches/runs
}

type roundInfo struct {
	roundNum  int
	items     []string
	converged bool
}

type roundPauseInfo struct {
	round                   int
	itemStatsMap            map[string]*itemStats
	seconds                 int
	completedIterations     int
	maxIterations           int
}

type RoundCutoff struct {
	Round        int
	Position     int    // Position in sorted list where cut was made
	Method       string // "MAX(stable)" or "MEDIAN(unstable)"
	ItemsBefore  int    // Number of items before cutoff
	ItemsAfter   int    // Number of items after cutoff
}

type ItemDisplayState struct {
	ID              string
	Value           string
	RoundEliminated int     // 0 if still active, otherwise elimination round
	LastRank        float64 // Last known average rank
	LastStdDev      float64 // Last known std dev
}

// Pipeline data structures
type batchJob struct {
	iteration int
	batch     int
	objects   []object
}

type batchResult struct {
	iteration     int
	batch         int
	rankedObjects []rankedObject
	err          error
}

// Helper methods to get round-specific parameters
func (r *Ranker) getThreshold(round int) float64 {
	if round == 1 {
		return r.cfg.ThreshStdDev
	}
	return r.cfg.RankStdDev
}

func (r *Ranker) getCutoffWindow(round int) int {
	if round == 1 {
		return r.cfg.ThreshCutoffWindow
	}
	return r.cfg.RankCutoffWindow
}

func (r *Ranker) getCutoffRange(round int) float64 {
	if round == 1 {
		return r.cfg.ThreshCutoffRange
	}
	return r.cfg.RankCutoffRange
}

func (r *Ranker) getStableRuns(round int) int {
	if round == 1 {
		return r.cfg.ThreshStableRuns
	}
	return r.cfg.RankStableRuns
}

// Helper function to calculate average standard deviation
func calculateAvgStdDev(itemStatsMap map[string]*itemStats) float64 {
	if len(itemStatsMap) == 0 {
		return 0.0
	}

	totalStdDev := 0.0
	for _, stats := range itemStatsMap {
		totalStdDev += stats.stdDev
	}
	return totalStdDev / float64(len(itemStatsMap))
}

// Helper function for elbow convergence detection (Round 1)
func (r *Ranker) checkConvergenceElbow(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int) (bool, int) {
	// Calculate current elbow position for logging purposes
	sortedStats := sortByAvgRank(itemStatsMap)

	var scores []float64
	if r.cfg.StdDevElbow {
		for _, stats := range sortedStats {
			scores = append(scores, stats.stdDev)
		}
	} else {
		for _, stats := range sortedStats {
			scores = append(scores, stats.avgRank)
		}
	}
	currentElbowIndex := detectElbow(scores)

	// Only start collecting elbow positions AFTER min-runs are completed
	if iteration <= minIterations {
		r.cfg.Logger.Debug("Convergence check - before/at min iterations, not collecting", "round", round,
			"iteration", iteration, "minIterations", minIterations, "currentElbow", currentElbowIndex)
		return false, currentElbowIndex
	}

	// Now we're past min-runs, start collecting elbow positions
	r.recentElbowPositions = append(r.recentElbowPositions, currentElbowIndex)
	cutoffWindow := r.getCutoffWindow(round)
	if len(r.recentElbowPositions) > cutoffWindow {
		r.recentElbowPositions = r.recentElbowPositions[1:]
	}

	// Don't calculate MAD until we have collected the full window size
	if len(r.recentElbowPositions) < cutoffWindow {
		r.consecutiveStableCount = 0
		r.cfg.Logger.Debug("Convergence check - insufficient window data", "round", round,
			"iteration", iteration, "currentElbow", currentElbowIndex,
			"collected", len(r.recentElbowPositions), "needWindow", cutoffWindow)
		return false, currentElbowIndex
	}

	// Calculate range of recent elbow positions
	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	// Use range threshold from config
	rangeThreshold := r.getCutoffRange(round) * float64(len(sortedStats))
	isStable := float64(elbowRange) <= rangeThreshold

	// Track consecutive stable iterations
	if isStable {
		r.consecutiveStableCount++
	} else {
		r.consecutiveStableCount = 0
	}

	// Require consecutive stable iterations (round-specific)
	stableRuns := r.getStableRuns(round)
	converged := r.consecutiveStableCount >= stableRuns

	var effectiveCutoff int
	if converged {
		// STABLE: Use maximum of stable window as the effective cutoff (most conservative)
		effectiveCutoff = maxInt(r.recentElbowPositions)
		r.lastStableCutoff = effectiveCutoff
		r.cfg.Logger.Debug("Convergence reached - using MAXIMUM cutoff", "round", round,
			"recentElbows", r.recentElbowPositions, "maxCutoff", effectiveCutoff)
	} else {
		// UNSTABLE: Use median for visualization
		medianValue := calculateMedian(convertIntToFloat(r.recentElbowPositions))
		effectiveCutoff = int(math.Round(medianValue))
	}

	r.cfg.Logger.Debug("Elbow convergence check", "round", round, "iteration", iteration,
		"currentElbow", currentElbowIndex, "windowSize", len(r.recentElbowPositions),
		"elbowRange", elbowRange, "rangeThreshold", rangeThreshold,
		"isStable", isStable, "consecutiveStable", r.consecutiveStableCount,
		"needStable", stableRuns, "converged", converged, "effectiveCutoff", effectiveCutoff)

	return converged, effectiveCutoff
}

func (r *Ranker) checkConvergenceSmallDataset(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int) (bool, int) {
	sortedStats := sortByAvgRank(itemStatsMap)
	cutoffIndex := len(sortedStats) / 2 // 50% cutoff for small datasets

	// Don't check convergence until minimum iterations
	if iteration < minIterations {
		return false, cutoffIndex
	}

	// Calculate average stddev for simple stability check
	currentAvgStdDev := calculateAvgStdDev(itemStatsMap)

	// Check for stability (need previous value to compare)
	if r.previousAvgStdDev > 0 {
		relativeChange := math.Abs(r.previousAvgStdDev-currentAvgStdDev) / r.previousAvgStdDev

		// Use a simpler threshold for small datasets
		stabilityThreshold := 0.05 // 5% change threshold for small datasets

		if relativeChange < stabilityThreshold {
			r.plateauStableCount++
			r.cfg.Logger.Debug("Small dataset stability detected", "round", round,
				"avgStdDev", currentAvgStdDev, "change", relativeChange,
				"stableCount", r.plateauStableCount)
		} else {
			r.plateauStableCount = 0
		}

		// Require consecutive stable iterations (use config stable runs)
		stableRuns := r.getStableRuns(round)
		converged := r.plateauStableCount >= stableRuns

		if converged {
			r.cfg.Logger.Info("Small dataset converged via stddev stability", "round", round,
				"avgStdDev", currentAvgStdDev, "iterations", iteration, "items", len(sortedStats))
		}

		r.previousAvgStdDev = currentAvgStdDev
		return converged, cutoffIndex
	}

	// First iteration with stddev tracking
	r.previousAvgStdDev = currentAvgStdDev
	return false, cutoffIndex
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
		cfg:                    config,
		encoding:               encoding,
		rng:                    rand.New(rand.NewSource(time.Now().UnixNano())),
		lastCutoffMethod:       "elbow",
		recentElbowPositions:   []int{},
		consecutiveStableCount: 0,
		lastStableCutoff:       0,
	}, nil
}

// Helper function to convert []int to []float64
func convertIntToFloat(ints []int) []float64 {
	floats := make([]float64, len(ints))
	for i, v := range ints {
		floats[i] = float64(v)
	}
	return floats
}

func calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

func calculateMAD(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	median := calculateMedian(values)

	deviations := make([]float64, len(values))
	for i, v := range values {
		deviations[i] = math.Abs(v - median)
	}

	return calculateMedian(deviations)
}

// Range calculation helper functions
func minInt(slice []int) int {
	if len(slice) == 0 {
		return 0
	}
	min := slice[0]
	for _, v := range slice[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func maxInt(slice []int) int {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, v := range slice[1:] {
		if v > max {
			max = v
		}
	}
	return max
}


func calculateStandardDeviation(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return math.Sqrt(variance)
}

func sortByAvgRank(itemStatsMap map[string]*itemStats) []*itemStats {
	var items []*itemStats
	for _, item := range itemStatsMap {
		items = append(items, item)
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].avgRank == items[j].avgRank {
			return items[i].ID < items[j].ID // Use ID as tiebreaker for stable sorting
		}
		return items[i].avgRank < items[j].avgRank
	})
	return items
}

func sortByStdDev(itemStatsMap map[string]*itemStats) []*itemStats {
	var items []*itemStats
	for _, item := range itemStatsMap {
		items = append(items, item)
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].stdDev == items[j].stdDev {
			return items[i].ID < items[j].ID // Use ID as tiebreaker for stable sorting
		}
		// Lower std dev = more stable = higher in list
		return items[i].stdDev < items[j].stdDev
	})
	return items
}

func (r *Ranker) hasSignal(items []*itemStats) (bool, float64) {
	if len(items) == 0 {
		return false, 0.0
	}

	stableCount := 0
	threshold := r.cfg.StdDevItem

	for _, stats := range items {
		if stats.stdDev <= threshold {
			stableCount++
		}
	}

	stablePercentage := float64(stableCount) / float64(len(items))
	hasSignal := stablePercentage > r.cfg.NoSignalThreshold

	return hasSignal, stablePercentage
}

func (r *Ranker) setupScreenSignalHandling() {
	// Handle Ctrl+C and other signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Handle keyboard events
	go func() {
		defer func() {
			if recover() != nil {
				// Screen was closed, exit gracefully
				r.cleanupAndExit()
			}
		}()
		for {
			if r.screen == nil {
				return
			}
			ev := r.screen.PollEvent()
			if ev == nil {
				return
			}
			switch ev := ev.(type) {
			case *tcell.EventKey:
				if ev.Key() == tcell.KeyCtrlC || ev.Key() == tcell.KeyEscape || ev.Rune() == 'q' {
					r.cfg.Logger.Info("Interrupted by user")
					r.cleanupAndExit()
				}
			case *tcell.EventResize:
				if r.screen != nil {
					r.screen.Sync()
				}
			}
		}
	}()

	// Handle OS signals
	go func() {
		<-sigChan
		r.cfg.Logger.Info("Interrupted by signal")
		r.cleanupAndExit()
	}()
}

func (r *Ranker) cleanupAndExit() {
	// Properly cleanup the screen to restore terminal
	if r.screen != nil {
		r.screen.Fini()
		r.screen = nil
	}

	// Reset signal handlers
	signal.Reset()

	// Force reset terminal to normal state
	fmt.Print("\033[?1049l") // Exit alternate screen
	fmt.Print("\033[?25h")   // Show cursor
	fmt.Print("\033[0m")     // Reset all attributes
	fmt.Print("\033c")       // Reset terminal (equivalent to 'reset' command)

	os.Exit(0)
}

func (r *Ranker) identifyEliteTier() []string {
	// Work backwards through rounds
	for i := len(r.rounds) - 1; i >= 0; i-- {
		if r.rounds[i].converged {
			// This round had clear ordering
			// Elite tier = next round (first that didn't converge)
			if i+1 < len(r.rounds) {
				return r.rounds[i+1].items
			}
			// All rounds converged, elite tier is just winner
			if len(r.rounds) > 0 && len(r.rounds[len(r.rounds)-1].items) > 0 {
				return r.rounds[len(r.rounds)-1].items[:1]
			}
		}
	}
	// No round converged (unlikely), elite tier is just winner
	if len(r.rounds) > 0 && len(r.rounds[len(r.rounds)-1].items) > 0 {
		return r.rounds[len(r.rounds)-1].items[:1]
	}
	return []string{}
}

func detectElbow(scores []float64) int {
	n := len(scores)
	if n < 3 {
		return n / 2
	}

	maxDist := 0.0
	elbowIdx := 0

	x1, y1 := 0.0, scores[0]
	x2, y2 := float64(n-1), scores[n-1]

	for i := 1; i < n-1; i++ {
		xi, yi := float64(i), scores[i]

		A := y2 - y1
		B := x1 - x2
		C := x2*y1 - x1*y2

		dist := math.Abs(A*xi+B*yi+C) / math.Sqrt(A*A+B*B)

		if dist > maxDist {
			maxDist = dist
			elbowIdx = i
		}
	}

	return elbowIdx
}

func (r *Ranker) determineCutoff(sortedStats []*itemStats, round int) int {
	if len(sortedStats) <= 2 {
		r.cfg.Logger.Debug("Cutoff: too few items", "round", round, "items", len(sortedStats))
		return len(sortedStats)
	}

	// Round 1: Always use elbow detection for threshing
	if round == 1 {
		r.cfg.Logger.Info("Using elbow detection (threshing)",
			"round", round,
			"items", len(sortedStats))
	} else {
		// Round 2+: Use hybrid approach based on dataset size for ranking
		smallDatasetThreshold := r.cfg.RankElbowThreshold

		// For small datasets in Round 2+, use fixed ratio
		if len(sortedStats) <= smallDatasetThreshold {
			cutoffIndex := int(float64(len(sortedStats)) * r.cfg.RankCutoffRatio)

			// Ensure at least 1 item selected, but not all
			if cutoffIndex < 1 {
				cutoffIndex = 1
			}
			if cutoffIndex >= len(sortedStats) {
				cutoffIndex = len(sortedStats) - 1
			}

			r.cfg.Logger.Info("Using fixed ratio cutoff (small dataset)",
				"round", round,
				"ratio", r.cfg.RankCutoffRatio,
				"cutoff", cutoffIndex,
				"items", len(sortedStats),
				"threshold", smallDatasetThreshold)

			return cutoffIndex
		}

		// For large datasets in Round 2+, use elbow detection
		r.cfg.Logger.Info("Using elbow detection (large dataset ranking)",
			"round", round,
			"items", len(sortedStats))
	}

	threshold := r.getThreshold(round)
	r.cfg.Logger.Info("Starting elbow cutoff determination", "round", round, "totalItems", len(sortedStats), "threshold", threshold)

	// Find top block (converged good items) - definitely kept
	topBlockSize := 0
	for i, stats := range sortedStats {
		if stats.stdDev <= threshold {
			topBlockSize++
			r.cfg.Logger.Debug("Top block item", "position", i, "id", stats.ID,
				"stdDev", stats.stdDev, "threshold", threshold)
		} else {
			break
		}
	}
	r.cfg.Logger.Info("Top block determined", "round", round, "size", topBlockSize)

	// Find bottom block (converged bad items, working backwards)
	// FIX: Only count items not already in top block
	bottomBlockSize := 0
	tolerance := r.cfg.BlockTolerance

	// Only look for bottom block if top block doesn't include everything
	if topBlockSize < len(sortedStats) {
		if tolerance == 0 {
			// Original contiguous behavior - but skip items in top block
			for i := len(sortedStats) - 1; i >= topBlockSize; i-- { // Start AFTER top block
				if sortedStats[i].stdDev <= threshold {
					bottomBlockSize++
				} else {
					break
				}
			}
		} else {
			// Tolerance-aware bottom block building - but skip items in top block
			consecutiveSkips := 0
			for i := len(sortedStats) - 1; i >= topBlockSize; i-- { // Start AFTER top block
				if sortedStats[i].stdDev <= threshold {
					bottomBlockSize++
					consecutiveSkips = 0
				} else {
					// Non-converged item - check if we can skip it
					if consecutiveSkips >= tolerance {
						// Already at or exceeded tolerance, stop building block
						break
					}

					// Look ahead (backwards) to see if there are more converged items within tolerance
					foundConverged := false
					lookAheadLimit := tolerance - consecutiveSkips
					// Ensure we don't look into the top block
					for j := i - 1; j >= i-lookAheadLimit && j >= topBlockSize; j-- {
						if sortedStats[j].stdDev <= threshold {
							foundConverged = true
							break
						}
					}

					if !foundConverged {
						// No converged items within remaining tolerance, stop
						break
					}

					// Include this non-converged item in the block size (we're skipping over its non-convergence)
					bottomBlockSize++
					consecutiveSkips++
				}
			}
		}
	}

	// Define analysis range: from start (including top block) to bottom block boundary
	analysisEnd := len(sortedStats) - bottomBlockSize

	// If bottom block reaches or passes top block, no analysis needed
	if analysisEnd <= topBlockSize {
		r.cfg.Logger.Debug("Bottom block meets top block, using top block only",
			"round", round, "topBlock", topBlockSize, "bottomBlock", bottomBlockSize)
		return topBlockSize
	}

	// Build scores from start to bottom block boundary (includes top block + middle)
	var scores []float64
	if r.cfg.StdDevElbow {
		// Use std dev scores for elbow detection
		for i := 0; i < analysisEnd; i++ {
			scores = append(scores, sortedStats[i].stdDev)
		}
		r.lastCutoffMethod = "stddev-elbow"
	} else {
		// Use rank scores (existing behavior)
		for i := 0; i < analysisEnd; i++ {
			scores = append(scores, sortedStats[i].avgRank)
		}
		r.lastCutoffMethod = "rank-elbow"
	}

	// Always use elbow detection
	cutoffIndex := detectElbow(scores)

	r.cfg.Logger.Debug("Using elbow detection", "round", round,
		"analysisRange", analysisEnd, "bottomBlockExcluded", bottomBlockSize,
		"cutoffIndex", cutoffIndex)

	// Ensure we keep at least the top block
	if cutoffIndex < topBlockSize {
		r.cfg.Logger.Debug("Adjusting cutoff to include all top block",
			"round", round, "curveCutoff", cutoffIndex, "topBlock", topBlockSize)
		cutoffIndex = topBlockSize
	}

	if cutoffIndex < 1 {
		cutoffIndex = 1
	}
	if cutoffIndex > len(sortedStats) {
		cutoffIndex = len(sortedStats)
	}

	r.cfg.Logger.Info("FINAL CUTOFF DECISION", "round", round,
		"totalItems", len(sortedStats),
		"topBlock", topBlockSize,
		"bottomBlock", bottomBlockSize,
		"analysisRange", analysisEnd,
		"elbowPosition", cutoffIndex,
		"finalCutoff", cutoffIndex,
		"willStopRecursion", cutoffIndex >= len(sortedStats))

	return cutoffIndex
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
	Object    object
	Score     float64
	Reasoning string
}

type rankedObjectResponse struct {
	Objects   []string          `json:"objects" jsonschema_description:"List of ranked object IDs"`
	Reasoning map[string]string `json:"reasoning,omitempty" jsonschema_description:"Brief reasoning for each object's position"`
}

type ReasoningProsCons struct {
	Pros string `json:"pros"` // Points that weighed in favor of this item
	Cons string `json:"cons"` // Points that weighed against this item
}

type RankedObject struct {
	Key       string              `json:"key"`
	Value     string              `json:"value"`
	Object    interface{}         `json:"object"` // if loading from json file
	Score     float64             `json:"score"`
	Exposure  int                 `json:"exposure"`
	Rank      int                 `json:"rank"`
	LastRound int                 `json:"last_round"` // 1-based: 1=threshing, 2+=ranking rounds
	Reasoning *ReasoningProsCons  `json:"reasoning,omitempty"`  // Pros and cons from comparisons
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

	// Also translate keys in the Reasoning map
	if response.Reasoning != nil {
		translatedReasoning := make(map[string]string)
		for tempID, reasoning := range response.Reasoning {
			if originalID, exists := tempToOriginal[tempID]; exists {
				translatedReasoning[originalID] = reasoning
			} else {
				translatedReasoning[tempID] = reasoning
			}
		}
		response.Reasoning = translatedReasoning
	}
}

var rankedObjectResponseSchema = generateSchema[rankedObjectResponse]()

func init() {
	// Debug: Print schema to verify reasoning field is included
	if false { // Set to true to debug schema
		b, _ := json.MarshalIndent(rankedObjectResponseSchema, "", "  ")
		fmt.Println("Ranked Object Response Schema:")
		fmt.Println(string(b))
	}
}

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

	// Initialize allItemStats for reasoning collection
	r.allItemStats = make(map[string]*itemStats)

	// Initialize screen at the top level if observe mode is enabled
	if r.cfg.Observe && r.screen == nil {
		// Redirect logger to file when in observe mode
		logFile, err := os.OpenFile("raink.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			r.cfg.Logger.Warn("Failed to create log file, continuing with stderr", "error", err)
		} else {
			// Create new logger that writes to file
			r.cfg.Logger = slog.New(slog.NewTextHandler(logFile, &slog.HandlerOptions{
				Level:     r.cfg.LogLevel,
				AddSource: false,
			})).With("component", "raink")

			// Ensure log file gets closed
			defer logFile.Close()

			r.cfg.Logger.Info("Observe mode enabled - logging to raink.log")
		}

		r.screen, err = tcell.NewScreen()
		if err != nil {
			r.cfg.Logger.Warn("Failed to create screen for visualization, disabling observe mode", "error", err)
			r.cfg.Observe = false
		} else {
			err = r.screen.Init()
			if err != nil {
				r.cfg.Logger.Warn("Failed to initialize screen for visualization, disabling observe mode", "error", err)
				r.cfg.Observe = false
				r.screen = nil
			} else {
				// Initialize progressive mode if enabled
				if r.cfg.ProgressiveViz {
					r.progressiveMode = true
					r.displayStates = make(map[string]*ItemDisplayState)
					r.allRoundStats = make(map[string]*itemStats)
					r.roundCutoffs = []RoundCutoff{}
					r.cfg.Logger.Info("Progressive visualization mode enabled")
				}

				// Initialize allRoundStats for minimap even in normal mode
				if r.cfg.MinimapEnabled && r.allRoundStats == nil {
					r.allRoundStats = make(map[string]*itemStats)
					r.cfg.Logger.Info("Minimap mode enabled - tracking complete dataset")
				}

				// Setup signal handling at this level
				r.setupScreenSignalHandling()
				defer func() {
					// Clean exit without os.Exit() since this is normal completion
					if r.screen != nil {
						r.screen.Fini()
						r.screen = nil
					}
					signal.Reset()
					// Restore terminal for normal completion
					fmt.Print("\033[?1049l") // Exit alternate screen
					fmt.Print("\033[?25h")   // Show cursor
					fmt.Print("\033[0m")     // Reset all attributes
				}()
			}
		}
	}

	// Track initial object IDs for round 1 elimination filtering
	initialIDs := make(map[string]bool)
	for _, obj := range objects {
		initialIDs[obj.ID] = true
	}

	// Clear rounds tracking
	r.rounds = []roundInfo{}

	results := r.rank(objects, 1)

	// Identify elite tier
	eliteTier := r.identifyEliteTier()
	eliteTierMap := make(map[string]bool)
	for _, id := range eliteTier {
		eliteTierMap[id] = true
	}

	// Track which round each item lasted through (1-based: 1=threshing, 2+=ranking rounds)
	itemLastRound := make(map[string]int)

	// Default: items that survive get the last round number
	lastRoundNum := len(r.rounds) // 1-based round number
	if lastRoundNum < 1 {
		lastRoundNum = 1
	}
	for _, result := range results {
		itemLastRound[result.Key] = lastRoundNum
	}

	// Walk through rounds to find when items were eliminated
	if len(r.rounds) > 1 {
		for i := 0; i < len(r.rounds)-1; i++ {
			// Build set of items that advanced to next round
			nextRoundItems := make(map[string]bool)
			for _, id := range r.rounds[i+1].items {
				nextRoundItems[id] = true
			}

			// Items in current round but not next round were eliminated
			for _, id := range r.rounds[i].items {
				if !nextRoundItems[id] && itemLastRound[id] == lastRoundNum {
					itemLastRound[id] = i + 1 // Convert 0-based array index to 1-based round number
				}
			}
		}
	}

	// Attach elimination info to all results
	for i := range results {
		results[i].LastRound = itemLastRound[results[i].Key]
	}

	filteredResults := results // No filtering - include everything

	// Add the rank key to each final result based on its position in the list
	for i := range filteredResults {
		filteredResults[i].Rank = i + 1
	}

	// Summarize reasoning if enabled
	if r.cfg.Reasoning {
		r.cfg.Logger.Info("Summarizing reasoning for all items", "count", len(filteredResults))

		// Parallelize summarization using goroutines
		type summaryJob struct {
			index    int
			key      string
			value    string
			snippets []string
		}

		type summaryResult struct {
			index   int
			summary *ReasoningProsCons
			err     error
		}

		jobs := make(chan summaryJob, len(filteredResults))
		results := make(chan summaryResult, len(filteredResults))

		// Start worker pool
		numWorkers := r.cfg.BatchConcurrency
		if numWorkers > len(filteredResults) {
			numWorkers = len(filteredResults)
		}

		for w := 0; w < numWorkers; w++ {
			go func(workerID int) {
				for job := range jobs {
					r.cfg.Logger.Info("Summarizing reasoning", "item", job.index+1, "of", len(filteredResults), "snippets", len(job.snippets), "worker", workerID)
					summary, err := r.summarizeReasoning(job.key, job.value, job.snippets)
					if err != nil {
						r.cfg.Logger.Warn("Failed to summarize reasoning", "item", job.key, "error", err)
					} else if summary != nil {
						prosLen := len(summary.Pros)
						consLen := len(summary.Cons)
						r.cfg.Logger.Info("Completed reasoning summary", "item", job.index+1, "of", len(filteredResults), "prosLen", prosLen, "consLen", consLen, "worker", workerID)
					}
					results <- summaryResult{index: job.index, summary: summary, err: err}
				}
			}(w)
		}

		// Queue all jobs
		itemsToSummarize := 0
		for i := range filteredResults {
			if stats, exists := r.allItemStats[filteredResults[i].Key]; exists && len(stats.reasoningSnippets) > 0 {
				jobs <- summaryJob{
					index:    i,
					key:      filteredResults[i].Key,
					value:    filteredResults[i].Value,
					snippets: stats.reasoningSnippets,
				}
				itemsToSummarize++
			} else {
				// No reasoning collected (e.g., eliminated in round 1)
				filteredResults[i].Reasoning = nil
			}
		}
		close(jobs)

		// Collect results
		for j := 0; j < itemsToSummarize; j++ {
			result := <-results
			if result.err != nil {
				filteredResults[result.index].Reasoning = nil
			} else {
				filteredResults[result.index].Reasoning = result.summary
			}
		}
		close(results)
	}

	// Log elite tier info
	if len(eliteTier) > 0 {
		r.cfg.Logger.Info("Elite tier identified", "items", eliteTier)
	}

	// Show final stop reason if in observe mode
	if r.cfg.Observe && r.screen != nil && r.finalStopReason != StopReasonNone {
		r.renderFinalStop(r.finalStopReason, r.round)
		if r.cfg.RoundPause > 0 {
			time.Sleep(time.Duration(r.cfg.RoundPause) * time.Second)
		}
	} else if r.cfg.Observe && r.cfg.RoundPause > 0 {
		// Fallback for when no specific stop reason was set
		r.cfg.Logger.Info("Final results displayed - pausing before exit", "seconds", r.cfg.RoundPause)
		time.Sleep(time.Duration(r.cfg.RoundPause) * time.Second)
	}

	// Clean up screen before returning so JSON output is clean
	if r.screen != nil {
		r.screen.Fini()
		r.screen = nil
	}

	return filteredResults, nil
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

	// DEBUG: Log items received at start of round
	var itemIDs []string
	for _, obj := range objects {
		itemIDs = append(itemIDs, obj.ID)
	}
	r.cfg.Logger.Info("ROUND START DEBUG",
		"round", round,
		"receivedItems", len(objects),
		"itemIDs", itemIDs)

	// Reset plateau tracking for new round
	r.previousAvgStdDev = 0
	r.plateauStableCount = 0

	roundName := "Ranking"
	if round == 1 {
		roundName = "Threshing"
	}
	r.cfg.Logger.Info(fmt.Sprintf("%s objects", roundName), "round", r.round, "count", len(objects))

	// Track round info for convergence tracking
	roundItems := make([]string, len(objects))
	for i, obj := range objects {
		roundItems[i] = obj.ID
	}

	// If we've narrowed down to a single object, we're done.
	if len(objects) == 1 {
		r.finalStopReason = StopReasonSingleItem
		r.rounds = append(r.rounds, roundInfo{
			roundNum:  round,
			items:     roundItems,
			converged: true,
		})

		// Show final stop visualization if in observe mode
		if r.cfg.Observe && r.screen != nil {
			r.renderFinalStop(StopReasonSingleItem, round)
		}

		return []RankedObject{
			{
				Key:      objects[0].ID,
				Value:    objects[0].Value,
				Object:   objects[0].Object,
				Score:    0,
				Exposure: 1,
				LastRound: 1,
			},
		}
	}

	// Adjust batch size if needed
	if r.cfg.BatchSize > len(objects) {
		r.cfg.BatchSize = len(objects)
	}

	// Run convergence-based iterations
	itemStatsMap := r.iterateUntilConverged(objects, round)

	// Check if no-signal was detected
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	// Check for convergence milestone: when no items have converged (all have high stdDev)
	if noSignalDetected {
		r.cfg.Logger.Info("Convergence milestone: items have similar relevance", "round", round)
		// Mark all items in this round as high-confidence group
		for _, stats := range itemStatsMap {
			stats.highConfidenceGroup = true
		}
		// Continue processing instead of stopping
	}

	// Handle any pending round pause immediately after iterations complete
	if r.pendingRoundPause != nil {
		r.cfg.Logger.Info("Handling pending round pause", "round", r.pendingRoundPause.round)
		// Just pause with the last displayed frame - don't render a special completion frame
		r.pauseWithVisualization(r.pendingRoundPause.round, r.pendingRoundPause.seconds)
		r.pendingRoundPause = nil // Clear the pending pause
	}

	// Check if we converged by simple threshold check
	converged := true
	threshold := r.getThreshold(round)
	for _, stats := range itemStatsMap {
		if stats.stdDev >= threshold {
			converged = false
			break
		}
	}

	r.rounds = append(r.rounds, roundInfo{
		roundNum:  round,
		items:     roundItems,
		converged: converged,
	})

	// Pause after round completes (before any recursion) - non-observe mode only
	if r.cfg.RoundPause > 0 && (!r.cfg.Observe || r.screen == nil) {
		r.cfg.Logger.Info("Round completed, pausing", "round", round, "seconds", r.cfg.RoundPause)
		time.Sleep(time.Duration(r.cfg.RoundPause) * time.Second)
	}

	// Sort by average rank
	sortedStats := sortByAvgRank(itemStatsMap)

	// Determine cutoff
	cutoffIndex := r.determineCutoff(sortedStats, round)

	// DEBUG: Log cutoff details
	r.cfg.Logger.Info("CUTOFF DEBUG",
		"round", round,
		"cutoffIndex", cutoffIndex,
		"totalItems", len(sortedStats),
		"willAdvance", cutoffIndex)

	topPortion := sortedStats[:cutoffIndex]
	bottomPortion := sortedStats[cutoffIndex:]

	// DEBUG: Log exact items in each portion
	var topIDs []string
	for _, stats := range topPortion {
		topIDs = append(topIDs, stats.ID)
	}
	var bottomIDs []string
	for _, stats := range bottomPortion {
		bottomIDs = append(bottomIDs, stats.ID)
	}
	r.cfg.Logger.Info("PORTION DEBUG",
		"round", round,
		"topCount", len(topPortion),
		"topIDs", topIDs,
		"bottomCount", len(bottomPortion),
		"bottomIDs", bottomIDs)

	// INVERT if flag is set
	if r.cfg.InvertCutoff {
		r.cfg.Logger.Info("INVERT DEBUG - SWAPPING PORTIONS",
			"round", round,
			"beforeTop", len(topPortion),
			"beforeBottom", len(bottomPortion))
		topPortion, bottomPortion = bottomPortion, topPortion
	}

	// DEBUG: Ensure cutoff index matches what's shown in visualization
	if cutoffIndex != len(topPortion) {
		r.cfg.Logger.Error("CUTOFF MISMATCH",
			"cutoffIndex", cutoffIndex,
			"actualTopPortion", len(topPortion))
	}

	r.cfg.Logger.Info("Cutoff determined", "round", round,
		"refining", len(topPortion), "keeping_as_is", len(bottomPortion))

	// DEBUG: Progressive mode debug
	if r.progressiveMode {
		r.cfg.Logger.Info("PROGRESSIVE DEBUG",
			"round", round,
			"displayStatesCount", len(r.displayStates),
			"allRoundStatsCount", len(r.allRoundStats))
	}

	// Track cutoff information for progressive mode
	// Track cutoffs for progressive mode OR minimap mode
	if r.progressiveMode || r.cfg.MinimapEnabled {
		cutoffMethod := "MAX(stable)"
		if r.consecutiveStableCount < r.getStableRuns(round) {
			cutoffMethod = "MEDIAN(unstable)"
		}

		// Track which items are being eliminated (only in progressive mode)
		if r.progressiveMode {
			for i := cutoffIndex; i < len(sortedStats); i++ {
				if state, exists := r.displayStates[sortedStats[i].ID]; exists {
					if state.RoundEliminated == 0 {
						state.RoundEliminated = round
						state.LastRank = sortedStats[i].avgRank
						state.LastStdDev = sortedStats[i].stdDev
					}
				}
			}
		}

		// Initialize roundCutoffs if needed (for normal mode with minimap)
		if r.roundCutoffs == nil {
			r.roundCutoffs = []RoundCutoff{}
		}

		r.roundCutoffs = append(r.roundCutoffs, RoundCutoff{
			Round:       round,
			Position:    cutoffIndex,
			Method:      cutoffMethod,
			ItemsBefore: cutoffIndex,
			ItemsAfter:  len(sortedStats) - cutoffIndex,
		})

		r.cfg.Logger.Debug("Tracked cutoff", "round", round,
			"cutoffIndex", cutoffIndex, "method", cutoffMethod,
			"eliminated", len(sortedStats)-cutoffIndex, "mode", map[bool]string{true: "progressive", false: "minimap"}[r.progressiveMode])
	}

	// New stop condition: unclear elbow indicates no meaningful distinction
	if cutoffIndex == 0 || cutoffIndex >= len(sortedStats) {
		r.finalStopReason = StopReasonNoMoreCuts
		r.cfg.Logger.Info("Stopping: elbow at edge indicates no clear distinction", "round", round, "cutoffIndex", cutoffIndex, "totalItems", len(sortedStats))

		// Show final stop visualization if in observe mode
		if r.cfg.Observe && r.screen != nil {
			r.renderFinalStop(StopReasonNoMoreCuts, round)
		}

		var results []RankedObject
		for i, stats := range sortedStats {
			results = append(results, RankedObject{
				Key:      stats.ID,
				Value:    stats.Value,
				Object:   stats.Object,
				Score:    stats.avgRank,
				Exposure: len(stats.rankHistory),
				Rank:     i + 1,
				LastRound: 1,
			})
		}
		return results
	}

	// Base case for recursion
	if len(topPortion) < 2 || len(topPortion) == len(objects) {
		// Determine specific stop reason
		if len(topPortion) < 2 {
			r.finalStopReason = StopReasonTooFewItems
		} else {
			r.finalStopReason = StopReasonNoMoreCuts
		}

		// Show final stop visualization if in observe mode
		if r.cfg.Observe && r.screen != nil {
			r.renderFinalStop(r.finalStopReason, round)
		}

		var results []RankedObject
		for i, stats := range sortedStats {
			results = append(results, RankedObject{
				Key:      stats.ID,
				Value:    stats.Value,
				Object:   stats.Object,
				Score:    stats.avgRank,
				Exposure: len(stats.rankHistory),
				Rank:     i + 1,
				LastRound: 1,
			})
		}
		return results
	}

	// Log top items being refined
	r.cfg.Logger.Debug("Top items being sent back into recursion:")
	for i, stats := range topPortion {
		r.cfg.Logger.Debug("Recursive item", "rank", i+1, "id", stats.ID, "avgRank", stats.avgRank, "variance", stats.stdDev)
	}

	// Convert top portion back to objects for recursion
	var topObjects []object
	for _, stats := range topPortion {
		topObjects = append(topObjects, object{
			ID:     stats.ID,
			Value:  stats.Value,
			Object: stats.Object,
		})
	}

	// DEBUG: Log items going into recursion
	var objectIDs []string
	for _, obj := range topObjects {
		objectIDs = append(objectIDs, obj.ID)
	}
	r.cfg.Logger.Info("RECURSION DEBUG",
		"round", round,
		"objectCount", len(topObjects),
		"objectIDs", objectIDs)

	// Recurse on top portion
	refinedTop := r.rank(topObjects, round+1)

	// Adjust scores by depth (inverted weight)
	for i := range refinedTop {
		refinedTop[i].Score /= float64(2 * round)
	}

	// Convert bottom portion to RankedObjects
	var bottomResults []RankedObject
	startRank := len(refinedTop) + 1
	for i, stats := range bottomPortion {
		bottomResults = append(bottomResults, RankedObject{
			Key:      stats.ID,
			Value:    stats.Value,
			Object:   stats.Object,
			Score:    stats.avgRank,
			Exposure: len(stats.rankHistory),
			Rank:     startRank + i,
			LastRound: 1,
		})
	}

	// Combine results
	finalResults := append(refinedTop, bottomResults...)
	return finalResults
}

func (r *Ranker) logFromApiCall(runNum, batchNum int, message string, args ...interface{}) {
	formattedMessage := fmt.Sprintf(message, args...)
	r.cfg.Logger.Debug(formattedMessage, "round", r.round, "run", runNum, "batch", batchNum, "total_batches", r.numBatches)
}


func (r *Ranker) buildBottomBlock(sortedItems []*itemStats, round int) []string {
	threshold := r.getThreshold(round)
	tolerance := r.cfg.BlockTolerance

	var bottomBlock []string

	if tolerance == 0 {
		// Original contiguous behavior
		for i := len(sortedItems) - 1; i >= 0; i-- {
			if sortedItems[i].stdDev <= threshold {
				bottomBlock = append(bottomBlock, sortedItems[i].ID)
			} else {
				break
			}
		}
		return bottomBlock
	}

	// Tolerance-aware bottom block building
	consecutiveSkips := 0
	for i := len(sortedItems) - 1; i >= 0; i-- {
		if sortedItems[i].stdDev <= threshold {
			// Converged item - add to block and reset skip counter
			bottomBlock = append(bottomBlock, sortedItems[i].ID)
			consecutiveSkips = 0
		} else {
			// Non-converged item - check if we can skip it
			if consecutiveSkips >= tolerance {
				// Already at or exceeded tolerance, stop building block
				break
			}

			// Look ahead (backwards) to see if there are more converged items within tolerance
			foundConverged := false
			lookAheadLimit := tolerance - consecutiveSkips
			for j := i - 1; j >= i-lookAheadLimit && j >= 0; j-- {
				if sortedItems[j].stdDev <= threshold {
					foundConverged = true
					break
				}
			}

			if !foundConverged {
				// No converged items within remaining tolerance, stop
				break
			}

			// Include this non-converged item in the block (we're skipping over its non-convergence)
			bottomBlock = append(bottomBlock, sortedItems[i].ID)
			consecutiveSkips++
		}
	}

	return bottomBlock
}

func (r *Ranker) checkConvergence(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int) (bool, int) {
	sortedStats := sortByAvgRank(itemStatsMap)

	if round == 1 {
		// Round 1: Always use elbow convergence detection for threshing
		return r.checkConvergenceElbow(itemStatsMap, round, iteration, minIterations)
	}

	// Round 2+: Use hybrid approach based on dataset size for ranking
	smallDatasetThreshold := r.cfg.RankElbowThreshold

	if len(sortedStats) <= smallDatasetThreshold {
		// Small dataset: Use simple plateau detection with fixed ratio
		cutoffIndex := int(float64(len(sortedStats)) * r.cfg.RankCutoffRatio)

		// Don't check convergence until minimum iterations
		if iteration < minIterations {
			return false, cutoffIndex
		}

		// Calculate average stddev for plateau detection
		currentAvgStdDev := calculateAvgStdDev(itemStatsMap)

		// Check for plateau (need previous value to compare)
		if r.previousAvgStdDev > 0 {
			relativeChange := math.Abs(r.previousAvgStdDev-currentAvgStdDev) / r.previousAvgStdDev

			if relativeChange < r.cfg.RankPlateauThreshold {
				r.plateauStableCount++
				r.cfg.Logger.Debug("Plateau detected (small dataset)", "round", round,
					"avgStdDev", currentAvgStdDev, "change", relativeChange,
					"stableCount", r.plateauStableCount)
			} else {
				r.plateauStableCount = 0
			}

			// Require consecutive stable iterations (reuse StableRuns config)
			stableRuns := r.getStableRuns(round)
			converged := r.plateauStableCount >= stableRuns

			if converged {
				r.cfg.Logger.Info("Small dataset ranking converged via plateau", "round", round,
					"avgStdDev", currentAvgStdDev, "iterations", iteration)
			}

			r.previousAvgStdDev = currentAvgStdDev
			return converged, cutoffIndex
		}

		// First iteration with stddev tracking
		r.previousAvgStdDev = currentAvgStdDev
		return false, cutoffIndex
	} else {
		// Large dataset: Use elbow convergence detection for ranking
		return r.checkConvergenceElbow(itemStatsMap, round, iteration, minIterations)
	}
}

func (r *Ranker) iterateUntilConverged(objects []object, round int) map[string]*itemStats {
	// Initialize item stats
	itemStatsMap := make(map[string]*itemStats)
	for _, obj := range objects {
		itemStatsMap[obj.ID] = &itemStats{
			ID: obj.ID, Value: obj.Value, Object: obj.Object, rankHistory: []float64{},
		}
	}

	n := len(objects)

	// Use config values if provided, otherwise use defaults
	var minIterations, maxIterations int
	if r.cfg.MinRuns > 0 {
		minIterations = r.cfg.MinRuns
	} else {
		minIterations = int(math.Ceil(math.Log2(float64(n))))
	}

	if r.cfg.MaxRuns > 0 {
		maxIterations = r.cfg.MaxRuns
	} else {
		maxIterations = int(math.Ceil(math.Sqrt(float64(n))))
	}

	r.numBatches = len(objects) / r.cfg.BatchSize

	// Create cancellable context for early termination
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Pipeline state - size channels based on total work
	totalBatches := r.numBatches * maxIterations
	batchQueueSize := totalBatches + 100  // Add some buffer
	resultQueueSize := r.cfg.BatchConcurrency * 2  // Buffer for in-flight results
	pendingBatches := make(chan batchJob, batchQueueSize)  // Queue of work to be done
	results := make(chan batchResult, resultQueueSize)     // Completed work
	iterationData := make(map[int]map[int][]rankedObject) // iter -> batch -> results
	completedIterations := make(map[int]bool)

	// Reset convergence tracking
	r.recentElbowPositions = []int{}
	r.consecutiveStableCount = 0

	r.cfg.Logger.Info("Starting TRUE pipeline", "round", round, "concurrency", r.cfg.BatchConcurrency, "minIter", minIterations, "maxIter", maxIterations, "totalBatches", totalBatches, "queueSize", batchQueueSize)

	// Log to console for immediate user feedback (bypass any redirected stderr)
	if r.cfg.Observe {
		consoleFd := os.NewFile(uintptr(2), "/dev/stderr") // Direct stderr access
		fmt.Fprintf(consoleFd, "[raink] Starting pipeline with %d workers, %d batches across %d iterations...\n",
			r.cfg.BatchConcurrency, r.numBatches*maxIterations, maxIterations)
	}

	// Fill initial work queue
	r.cfg.Logger.Info("Queueing all batches", "totalBatches", totalBatches)
	r.fillWorkQueue(pendingBatches, objects, 1, maxIterations)

	// Launch worker pool with context - these run continuously
	for i := 0; i < r.cfg.BatchConcurrency; i++ {
		go r.pipelineWorker(i, ctx, pendingBatches, results)
	}

	// Log worker startup
	if r.cfg.Observe {
		consoleFd := os.NewFile(uintptr(2), "/dev/stderr")
		fmt.Fprintf(consoleFd, "[raink] %d workers launched, processing batches...\n", r.cfg.BatchConcurrency)
	}

	// Process results and manage convergence
	return r.processPipelineResults(ctx, cancel, results, iterationData, completedIterations, itemStatsMap, round, minIterations, maxIterations, pendingBatches)
}

// Pipeline worker pool
func (r *Ranker) pipelineWorker(workerID int, ctx context.Context, jobs <-chan batchJob, results chan<- batchResult) {
	for {
		select {
		case <-ctx.Done():
			r.cfg.Logger.Debug("Worker cancelled", "worker", workerID, "reason", ctx.Err())
			return
		case job, ok := <-jobs:
			if !ok {
				r.cfg.Logger.Debug("Worker finished - job queue closed", "worker", workerID)
				return
			}

			// Check if we should skip this job due to cancellation
			if ctx.Err() != nil {
				r.cfg.Logger.Debug("Skipping job due to cancellation",
					"worker", workerID, "iteration", job.iteration, "batch", job.batch)
				continue
			}

			r.cfg.Logger.Debug("Worker processing batch", "worker", workerID, "iteration", job.iteration, "batch", job.batch)

			rankedBatch, err := r.rankObjects(job.objects, job.iteration, job.batch)

			// Check again if cancelled before sending result
			if ctx.Err() != nil {
				r.cfg.Logger.Debug("Discarding result due to cancellation",
					"worker", workerID, "iteration", job.iteration, "batch", job.batch)
				continue
			}

			results <- batchResult{
				iteration:     job.iteration,
				batch:         job.batch,
				rankedObjects: rankedBatch,
				err:          err,
			}
		}
	}
}

// Work queue management
func (r *Ranker) fillWorkQueue(queue chan<- batchJob, objects []object, startIteration, maxIterations int) {
	for iteration := startIteration; iteration <= maxIterations; iteration++ {
		// Shuffle objects for this iteration
		shuffledObjects := make([]object, len(objects))
		copy(shuffledObjects, objects)
		r.rng.Shuffle(len(shuffledObjects), func(i, j int) {
			shuffledObjects[i], shuffledObjects[j] = shuffledObjects[j], shuffledObjects[i]
		})

		// Create batches for this iteration
		for batch := 1; batch <= r.numBatches; batch++ {
			start := (batch - 1) * r.cfg.BatchSize
			end := batch * r.cfg.BatchSize
			batchObjects := shuffledObjects[start:end]

			// Add to work queue (non-blocking)
			select {
			case queue <- batchJob{iteration: iteration, batch: batch, objects: batchObjects}:
				r.cfg.Logger.Debug("Queued batch", "iteration", iteration, "batch", batch)
			default:
				r.cfg.Logger.Debug("Queue full, batch queued later", "iteration", iteration, "batch", batch)
				queue <- batchJob{iteration: iteration, batch: batch, objects: batchObjects} // blocking
			}
		}
	}

	r.cfg.Logger.Info("All batches queued", "totalIterations", maxIterations, "batchesPerIteration", r.numBatches)
}

// Results processing with early termination
func (r *Ranker) processPipelineResults(ctx context.Context, cancel context.CancelFunc, results <-chan batchResult, iterationData map[int]map[int][]rankedObject,
	completedIterations map[int]bool, itemStatsMap map[string]*itemStats, round, minIterations, maxIterations int,
	pendingBatches chan batchJob) map[string]*itemStats {

	expectedResults := maxIterations * r.numBatches
	processedResults := 0
	convergedAt := -1
	totalCompletedIterations := 0
	firstResultReceived := false

	for result := range results {
		if result.err != nil {
			r.cfg.Logger.Error("Batch error", "error", result.err)
			continue
		}

		// Log when first result comes in
		if !firstResultReceived && r.cfg.Observe {
			consoleFd := os.NewFile(uintptr(2), "/dev/stderr")
			fmt.Fprintf(consoleFd, "[raink] First batch completed, visualization starting soon...\n")
			firstResultReceived = true
		}

		// Store result
		if iterationData[result.iteration] == nil {
			iterationData[result.iteration] = make(map[int][]rankedObject)
		}
		iterationData[result.iteration][result.batch] = result.rankedObjects

		processedResults++
		r.cfg.Logger.Debug("Batch completed", "iteration", result.iteration, "batch", result.batch, "processed", processedResults, "expected", expectedResults)

		// Check if iteration is complete
		if len(iterationData[result.iteration]) == r.numBatches && !completedIterations[result.iteration] {
			completedIterations[result.iteration] = true
			totalCompletedIterations++
			r.updateStatsFromIteration(result.iteration, iterationData[result.iteration], itemStatsMap)

			r.cfg.Logger.Info("Iteration completed", "iteration", result.iteration, "totalCompleted", totalCompletedIterations, "round", round, "minIterations", minIterations, "windowSize", len(r.recentElbowPositions))

			// Check convergence after minimum iterations
			if result.iteration >= minIterations && convergedAt == -1 {
				r.cfg.Logger.Debug("Checking convergence", "iteration", result.iteration, "windowSize", len(r.recentElbowPositions), "minIterations", minIterations)
				if r.checkConvergencePipeline(itemStatsMap, round) {
					// consecutiveStableCount is updated inside checkConvergencePipeline
					if r.consecutiveStableCount >= r.getStableRuns(round) {
						convergedAt = result.iteration
						r.cfg.Logger.Info("CONVERGENCE DETECTED - cancelling remaining work", "round", round,
							"iteration", result.iteration, "stableCount", r.consecutiveStableCount)

						// Check for no-signal condition after convergence is detected
						if r.cfg.StopOnNoSignal {
							// Determine what portion would be selected
							sortedStats := sortByAvgRank(itemStatsMap)
							cutoffIndex := r.determineCutoff(sortedStats, round)

							var selectedPortion []*itemStats
							if r.cfg.InvertCutoff {
								selectedPortion = sortedStats[cutoffIndex:]
							} else {
								selectedPortion = sortedStats[:cutoffIndex]
							}

							hasSignal, stablePercent := r.hasSignal(selectedPortion)

							if !hasSignal {
								r.cfg.Logger.Info("NO SIGNAL DETECTED - cancelling immediately",
									"round", round,
									"cutoff", cutoffIndex,
									"stablePercent", fmt.Sprintf("%.1f%%", stablePercent*100),
									"threshold", fmt.Sprintf("%.1f%%", r.cfg.NoSignalThreshold*100))

								// Mark for special handling in rank()
								for _, stats := range itemStatsMap {
									stats.noSignal = true
								}

								// Cancel immediately for no-signal case
								jobsInQueue := len(pendingBatches)
								r.savedAPICalls += jobsInQueue
								cancel()
								close(pendingBatches)
								expectedResults = processedResults + r.cfg.BatchConcurrency

								r.cfg.Logger.Info("NO SIGNAL - saved API calls",
									"saved", jobsInQueue,
									"totalSaved", r.savedAPICalls)
								break // Exit the loop immediately
							}
						} else {
							// Normal convergence case (has signal)
							// Count jobs remaining in queue before cancellation
							jobsInQueue := len(pendingBatches)
							r.savedAPICalls += jobsInQueue

							// Cancel all workers to stop processing
							cancel()

							// Close the work queue to stop new work
							close(pendingBatches)

							// Only wait for currently processing batches
							expectedResults = processedResults + r.cfg.BatchConcurrency

							r.cfg.Logger.Info("Saved API calls by convergence",
								"saved", jobsInQueue,
								"totalSaved", r.savedAPICalls,
								"processedSoFar", processedResults,
								"expectedAfterCancel", expectedResults)
						}
					}
				}
				// Note: consecutiveStableCount reset happens inside checkConvergencePipeline when not converged
			}

			// Visualization update
			if r.cfg.Observe && r.screen != nil {
				if totalCompletedIterations == 1 {
					consoleFd := os.NewFile(uintptr(2), "/dev/stderr")
					fmt.Fprintf(consoleFd, "[raink] Visualization now active!\n")
				}
				if r.progressiveMode {
					r.renderProgressiveVisualizationWithOptionalMinimap(itemStatsMap, round, totalCompletedIterations, minIterations, maxIterations)
				} else {
					r.renderVisualizationWithOptionalMinimap(itemStatsMap, round, totalCompletedIterations, minIterations, maxIterations)
				}
				if r.cfg.RunPause > 0 {
					time.Sleep(time.Duration(r.cfg.RunPause) * time.Second)
				}
			}
		}

		// Exit when we've processed enough results
		if processedResults >= expectedResults || (convergedAt != -1 && processedResults >= convergedAt*r.numBatches+r.cfg.BatchConcurrency) {
			r.cfg.Logger.Info("Pipeline processing complete", "processed", processedResults, "convergedAt", convergedAt)

			// If we hit max iterations without converging, log it
			if totalCompletedIterations >= maxIterations && convergedAt == -1 {
				// The effectiveCutoff will already be using median from checkConvergence
				r.cfg.Logger.Warn("Max iterations reached without convergence - using MEDIAN of elbow positions",
					"round", round,
					"completedIterations", totalCompletedIterations,
					"maxIterations", maxIterations,
					"recentElbows", r.recentElbowPositions,
					"medianElbow", int(calculateMedian(convertIntToFloat(r.recentElbowPositions))))
			}
			break
		}
	}

	// Mark that we want to pause after this round (will be handled by caller)
	if r.cfg.RoundPause > 0 && r.cfg.Observe && r.screen != nil {
		r.pendingRoundPause = &roundPauseInfo{
			round:               round,
			itemStatsMap:        itemStatsMap,
			seconds:             r.cfg.RoundPause,
			completedIterations: totalCompletedIterations,
			maxIterations:       maxIterations,
		}
	}

	// Log final API call savings summary
	if r.savedAPICalls > 0 {
		r.cfg.Logger.Info("Early termination summary",
			"totalSavedAPICalls", r.savedAPICalls,
			"finalProcessedResults", processedResults,
			"round", round)
	}

	// Copy stats to allItemStats for later summarization
	for id, stats := range itemStatsMap {
		r.allItemStats[id] = stats
	}

	return itemStatsMap
}

// Update statistics from completed iteration
func (r *Ranker) updateStatsFromIteration(iteration int, batchData map[int][]rankedObject, itemStatsMap map[string]*itemStats) {
	// Aggregate all batches from this iteration
	for batch := 1; batch <= r.numBatches; batch++ {
		if batchResults, exists := batchData[batch]; exists {
			for _, rankedObj := range batchResults {
				if stats, ok := itemStatsMap[rankedObj.Object.ID]; ok {
					stats.rankHistory = append(stats.rankHistory, rankedObj.Score)

					// Store reasoning if available
					if r.cfg.Reasoning {
						r.cfg.Logger.Debug("Reasoning check", "item", rankedObj.Object.ID, "round", r.round, "reasoningEnabled", r.cfg.Reasoning, "roundCheck", r.round > 1, "hasReasoning", rankedObj.Reasoning != "", "reasoningLen", len(rankedObj.Reasoning))
						if r.round > 1 && rankedObj.Reasoning != "" {
							r.cfg.Logger.Debug("Storing reasoning snippet", "item", rankedObj.Object.ID, "round", r.round, "snippet", rankedObj.Reasoning)
							stats.reasoningSnippets = append(stats.reasoningSnippets, rankedObj.Reasoning)
						}
					}
				}
			}
		}
	}

	// Update running averages and standard deviations
	for _, stats := range itemStatsMap {
		if len(stats.rankHistory) > 0 {
			sum := 0.0
			for _, rank := range stats.rankHistory {
				sum += rank
			}
			stats.avgRank = sum / float64(len(stats.rankHistory))
			stats.prevStdDev = stats.stdDev
			stats.stdDev = calculateStandardDeviation(stats.rankHistory)
		}
	}

	// Update elbow tracking
	sortedStats := sortByAvgRank(itemStatsMap)
	var scores []float64
	if r.cfg.StdDevElbow {
		for _, stats := range sortedStats {
			scores = append(scores, stats.stdDev)
		}
	} else {
		for _, stats := range sortedStats {
			scores = append(scores, stats.avgRank)
		}
	}

	elbowPosition := detectElbow(scores)
	r.recentElbowPositions = append(r.recentElbowPositions, elbowPosition)
	if len(r.recentElbowPositions) > r.cfg.CutoffWindow {
		r.recentElbowPositions = r.recentElbowPositions[1:]
	}

	r.cfg.Logger.Debug("Stats updated", "iteration", iteration, "elbow", elbowPosition, "windowSize", len(r.recentElbowPositions))
}

// Pipeline convergence check
func (r *Ranker) checkConvergencePipeline(itemStatsMap map[string]*itemStats, round int) bool {
	if len(r.recentElbowPositions) < 2 {
		r.cfg.Logger.Debug("Not enough elbow data for convergence check", "windowSize", len(r.recentElbowPositions))
		return false // Need minimum data
	}

	// Calculate range of recent elbow positions
	elbowMin := minInt(r.recentElbowPositions)
	elbowMax := maxInt(r.recentElbowPositions)
	elbowRange := elbowMax - elbowMin

	// Calculate threshold as percentage of dataset size
	datasetSize := len(itemStatsMap)
	rangeThreshold := r.getCutoffRange(round) * float64(datasetSize)
	converged := float64(elbowRange) <= rangeThreshold

	if converged {
		r.consecutiveStableCount++ // Update the display counter
		// Use maximum of stable window as effective cutoff
		r.lastStableCutoff = maxInt(r.recentElbowPositions)
	} else {
		r.consecutiveStableCount = 0 // Reset if not converged
	}

	r.cfg.Logger.Debug("Pipeline convergence check", "round", round,
		"recentElbows", r.recentElbowPositions,
		"elbowRange", elbowRange,
		"rangeThreshold", rangeThreshold,
		"datasetSize", datasetSize,
		"converged", converged,
		"consecutiveStable", r.consecutiveStableCount)

	return converged
}

func (r *Ranker) renderVisualizationWithOptionalMinimap(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if !r.cfg.MinimapEnabled || !r.cfg.Observe || r.screen == nil {
		r.renderVisualization(itemStatsMap, round, iteration, minIterations, maxIterations)
		return
	}

	r.renderWithMinimap(itemStatsMap, round, iteration, minIterations, maxIterations)
}

func (r *Ranker) renderWithMinimap(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if r.screen == nil {
		return
	}

	r.screen.Clear()
	width, height := r.screen.Size()

	// Calculate layout dimensions
	mainWidth := int(float64(width) * 0.8)
	minimapStartX := mainWidth + 1
	minimapWidth := width - minimapStartX - 1

	// Must have at least 5 chars for minimap to be useful
	if minimapWidth < 5 {
		// Fall back to full-width display
		r.renderVisualization(itemStatsMap, round, iteration, minIterations, maxIterations)
		return
	}

	// Draw vertical separator
	separatorStyle := tcell.StyleDefault.Foreground(tcell.ColorDarkGray)
	for y := 0; y < height; y++ {
		r.screen.SetContent(mainWidth, y, '│', nil, separatorStyle)
	}

	// Render main display (constrained to left portion)
	r.renderMainDisplayConstrained(itemStatsMap, round, iteration, minIterations, maxIterations, mainWidth, height)

	// Render minimap on right
	r.renderMinimap(itemStatsMap, minimapStartX, minimapWidth, height, round)

	r.screen.Show()
}

func (r *Ranker) renderMainDisplayConstrained(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int, maxWidth int, maxHeight int) {
	// This is mostly the same as renderVisualization but with width constraints

	// Update allRoundStats for minimap support (even in normal mode)
	// In normal mode, itemStatsMap only contains active items for this round,
	// so we only update active items, preserving eliminated items' frozen stats
	if r.cfg.MinimapEnabled && r.allRoundStats != nil {
		for id, stats := range itemStatsMap {
			r.allRoundStats[id] = stats
		}
	}

	// Sort items by appropriate metric
	var sortedItems []*itemStats
	var displayMode string
	if r.cfg.StdDevElbow {
		// Sort by std dev (most stable first)
		sortedItems = sortByStdDev(itemStatsMap)
		displayMode = "STABILITY MODE"
	} else {
		// Sort by average rank (existing)
		sortedItems = sortByAvgRank(itemStatsMap)
		displayMode = "RANK MODE"
	}

	// Determine cutoff for visualization
	cutoffIndex := r.determineCutoff(sortedItems, round)

	// Calculate debug info
	threshold := r.getThreshold(round)
	topBlockSize := 0
	for _, stats := range sortedItems {
		if stats.stdDev <= threshold {
			topBlockSize++
		} else {
			break
		}
	}

	bottomBlockSize := 0
	if topBlockSize < len(sortedItems) {
		for i := len(sortedItems) - 1; i >= topBlockSize; i-- {
			if sortedItems[i].stdDev <= threshold {
				bottomBlockSize++
			} else {
				break
			}
		}
	}
	if bottomBlockSize > len(sortedItems) - topBlockSize {
		bottomBlockSize = len(sortedItems) - topBlockSize
	}

	analysisEnd := len(sortedItems) - bottomBlockSize

	// Calculate current elbow index for display
	var scores []float64
	for _, stats := range sortedItems {
		scores = append(scores, stats.avgRank)
	}
	currentElbowIndex := detectElbow(scores)

	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	rangeThreshold := r.cfg.CutoffRangePercent * float64(len(sortedItems))

	// Check for no-signal condition
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	// Determine cutoff method for display
	var cutoffMethod string
	if r.consecutiveStableCount >= r.cfg.StableRuns {
		cutoffMethod = "MAX(stable)"
	} else {
		cutoffMethod = "MEDIAN(unstable)"
	}

	// Enhanced header with debug info
	var headerStr string
	if noSignalDetected {
		headerStr = fmt.Sprintf("%s | Round: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d | NO SIGNAL - NOISE DETECTED",
			displayMode, round, iteration, maxIterations,
			len(sortedItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	} else {
		headerStr = fmt.Sprintf("%s | Round: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d",
			displayMode, round, iteration, maxIterations,
			len(sortedItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	}

	// Truncate header if needed
	if len(headerStr) > maxWidth {
		headerStr = headerStr[:maxWidth-3] + "..."
	}
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorWhite))

	// Add explanation when cutoff equals total items (only after min iterations)
	startRow := 2
	if cutoffIndex >= len(sortedItems) && iteration >= minIterations {
		warningStr := fmt.Sprintf("WARNING: Cutoff=%d equals total items=%d - will select all, triggering stop condition",
			cutoffIndex, len(sortedItems))
		if len(warningStr) > maxWidth {
			warningStr = warningStr[:maxWidth-3] + "..."
		}
		r.writeString(0, 2, warningStr, tcell.StyleDefault.Foreground(tcell.ColorRed))
		startRow = 3
	}

	// Render help text
	helpStr := "Press Ctrl+C, Esc, or 'q' to quit"
	if len(helpStr) > maxWidth {
		helpStr = helpStr[:maxWidth]
	}
	r.writeString(0, startRow-1, helpStr, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))

	// Render each item with constrained width
	maxBarLength := (maxWidth - 20) / 2  // Leave room for ID and values
	if maxBarLength < 5 {
		maxBarLength = 5
	}
	centerX := maxWidth / 2

	for i, item := range sortedItems {
		if i >= maxHeight-(startRow+1) {
			break // Don't render beyond screen height
		}

		// Leave space for cutoff line - items at cutoffIndex and beyond get pushed down one row
		row := i + startRow
		if i >= cutoffIndex && cutoffIndex > 0 && cutoffIndex < len(sortedItems) {
			row += 1  // Skip the row reserved for cutoff line
		}

		// Determine color based on elbow position and convergence
		isAboveElbow := i < currentElbowIndex
		isConverged := item.stdDev <= threshold

		var style tcell.Style
		if isAboveElbow && isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
		} else if isAboveElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorYellow)
		} else if isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorBlue)
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorWhite)
		}

		// Calculate bar lengths (scaled to fit)
		stdDevBarLen := int(math.Min(float64(item.stdDev*10), float64(maxBarLength)))
		scoreBarLen := int(math.Max(0, float64(maxBarLength)-item.avgRank*10))
		if scoreBarLen > maxBarLength {
			scoreBarLen = maxBarLength
		}

		// Render std dev value
		stdDevStr := fmt.Sprintf("%.1f", item.stdDev)
		barStartX := centerX - len(item.ID)/2 - 2
		stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
		if stdDevX < 0 {
			stdDevX = 0
		}
		r.writeString(stdDevX, row, stdDevStr, style)

		// Render std dev bar (left side) with = characters only
		leftBarStart := centerX - len(item.ID)/2 - stdDevBarLen - 1
		for j := 0; j < stdDevBarLen && leftBarStart + j >= 0; j++ {
			r.screen.SetContent(leftBarStart + j, row, '=', nil, style)
		}

		// Render data content in right bar (score bar space)
		dataContent := item.Value
		rightBarStart := centerX + len(item.ID)/2 + 1
		for j := 0; j < scoreBarLen && rightBarStart + j < maxWidth; j++ {
			var ch rune
			if j < len(dataContent) {
				ch = rune(dataContent[j])
			} else {
				ch = '='
			}
			r.screen.SetContent(rightBarStart + j, row, ch, nil, style)
		}

		// Render item ID at center
		idX := centerX - len(item.ID)/2
		if idX >= 0 && idX + len(item.ID) < maxWidth {
			r.writeString(idX, row, item.ID, style)
		}

		// Render score value
		scoreStr := fmt.Sprintf("%.1f", item.avgRank)
		scoreX := centerX + len(item.ID)/2 + scoreBarLen + 3
		if scoreX + len(scoreStr) < maxWidth {
			r.writeString(scoreX, row, scoreStr, style)
		}
	}

	// Draw cutoff line if there's a meaningful cutoff
	if cutoffIndex > 0 && cutoffIndex < len(sortedItems) && cutoffIndex < maxHeight-(startRow+1) {
		// DEBUG: Log what's being displayed in visualization
		r.cfg.Logger.Info("VIZ CUTOFF DEBUG",
			"round", round,
			"cutoffIndex", cutoffIndex,
			"totalItems", len(sortedItems),
			"inverted", r.cfg.InvertCutoff)

		cutoffRow := cutoffIndex + startRow
		if cutoffRow < maxHeight-1 {
			cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
			if r.cfg.InvertCutoff {
				cutoffStyle = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
			}
			for x := 0; x < maxWidth; x++ {
				r.screen.SetContent(x, cutoffRow, '─', nil, cutoffStyle)
			}

			var cutoffMsg string
			if r.cfg.InvertCutoff {
				cutoffMsg = fmt.Sprintf(" INVERTED CUTOFF: Bottom %d advance to next round ", len(sortedItems)-cutoffIndex)
			} else {
				cutoffMsg = fmt.Sprintf(" CUTOFF: Top %d advance to next round ", cutoffIndex)
			}
			if len(cutoffMsg) > maxWidth {
				cutoffMsg = cutoffMsg[:maxWidth]
			}
			msgX := (maxWidth - len(cutoffMsg)) / 2
			if msgX >= 0 {
				r.writeString(msgX, cutoffRow, cutoffMsg, cutoffStyle)
			}
		}
	}
}

func (r *Ranker) renderMinimap(itemStatsMap map[string]*itemStats, startX int, width int, height int, round int) {
	if width < 5 {
		return // Not enough space
	}

	// Use current round data by default, complete dataset only with MinimapShowAll flag
	var completeStats map[string]*itemStats
	var activeItems []*itemStats
	var eliminatedItems []*itemStats

	if r.cfg.MinimapShowAll && len(r.allRoundStats) > 0 {
		// Use the complete dataset that includes all items from all rounds
		completeStats = r.allRoundStats

		// Separate active items from eliminated items for proper positioning
		// Active items are those in the current round's itemStatsMap
		for _, stats := range completeStats {
			if _, isActive := itemStatsMap[stats.ID]; isActive {
				activeItems = append(activeItems, stats)
			} else {
				eliminatedItems = append(eliminatedItems, stats)
			}
		}
	} else {
		// Default behavior: show only current round data
		completeStats = itemStatsMap

		// All items are active in current-round-only mode
		for _, stats := range completeStats {
			activeItems = append(activeItems, stats)
		}
		// No eliminated items to show
	}

	// Sort active items by current rank (they can shift around)
	if len(activeItems) > 0 {
		activeMap := make(map[string]*itemStats)
		for _, stats := range activeItems {
			activeMap[stats.ID] = stats
		}
		activeItems = sortByAvgRank(activeMap)
	}

	// Sort eliminated items by their frozen ranks (preserve elimination order)
	if len(eliminatedItems) > 0 {
		eliminatedMap := make(map[string]*itemStats)
		for _, stats := range eliminatedItems {
			eliminatedMap[stats.ID] = stats
		}
		eliminatedItems = sortByAvgRank(eliminatedMap)
	}

	// Combine: active items first, then eliminated items (preserves their frozen positions)
	var sortedStats []*itemStats
	sortedStats = append(sortedStats, activeItems...)
	sortedStats = append(sortedStats, eliminatedItems...)
	totalItems := len(sortedStats)

	// Render minimap header
	headerStr := fmt.Sprintf("Map:%d", totalItems)
	if len(headerStr) > width {
		headerStr = fmt.Sprintf("%d", totalItems)
	}
	headerStyle := tcell.StyleDefault.Foreground(tcell.ColorDarkGray)
	r.writeString(startX + (width-len(headerStr))/2, 0, headerStr, headerStyle)

	// Calculate display area
	displayStartRow := 2
	displayHeight := height - 3  // Leave room for header and bottom margin

	if displayHeight < 5 {
		return // Not enough vertical space
	}

	// Calculate compression ratio
	// When there are fewer items than display height, pack them toward the top
	var itemsPerRow float64
	var actualDisplayHeight int
	if totalItems <= displayHeight {
		// Compact layout: one item per row, packed toward top
		itemsPerRow = 1.0
		actualDisplayHeight = totalItems
	} else {
		// Normal layout: distribute items across available height
		itemsPerRow = float64(totalItems) / float64(displayHeight)
		actualDisplayHeight = displayHeight
	}

	// Find elbow position using only active items (current round)
	var scores []float64
	if r.cfg.StdDevElbow {
		for _, stats := range activeItems {
			scores = append(scores, stats.stdDev)
		}
	} else {
		for _, stats := range activeItems {
			scores = append(scores, stats.avgRank)
		}
	}
	elbowIndex := detectElbow(scores)

	// Create display buckets using actual display height
	buckets := make([]MinimapBucket, actualDisplayHeight)
	threshold := r.getThreshold(round)

	// Count eliminated items in each bucket
	eliminatedCounts := make([]int, actualDisplayHeight)
	activeCounts := make([]int, actualDisplayHeight)

	// Track the boundary between active and eliminated items
	activeBoundary := len(activeItems)

	for i, stats := range sortedStats {
		bucketIdx := int(float64(i) / itemsPerRow)
		if bucketIdx >= len(buckets) {
			bucketIdx = len(buckets) - 1
		}

		if buckets[bucketIdx].ItemCount == 0 {
			buckets[bucketIdx].StartIdx = i
		}
		buckets[bucketIdx].EndIdx = i
		buckets[bucketIdx].ItemCount++
		buckets[bucketIdx].AvgRank += stats.avgRank
		buckets[bucketIdx].AvgStdDev += stats.stdDev

		if stats.stdDev <= threshold {
			buckets[bucketIdx].ConvergedCount++
		}

		// Elbow is only within active items (first part of sortedStats)
		if i == elbowIndex && i < len(activeItems) {
			buckets[bucketIdx].ContainsElbow = true
		}

		// Track active vs eliminated items in each bucket
		isActive := i < activeBoundary
		if isActive {
			activeCounts[bucketIdx]++
		} else {
			eliminatedCounts[bucketIdx]++
		}

		// Note: Active/eliminated counting is now handled above using activeBoundary
	}

	// Normalize averages and render
	maxBarWidth := width - 2
	if maxBarWidth < 1 {
		maxBarWidth = 1
	}

	// Calculate min/max ranks for normalization
	var minRank, maxRank float64
	if totalItems > 0 {
		minRank = sortedStats[0].avgRank
		maxRank = sortedStats[len(sortedStats)-1].avgRank
	}

	// Calculate where to draw the active/eliminated separator
	activeSeparatorBucket := -1
	if len(eliminatedItems) > 0 && len(activeItems) > 0 {
		// Find the bucket where the transition from active to eliminated items occurs
		activeSeparatorBucket = int(float64(activeBoundary) / itemsPerRow)
		if activeSeparatorBucket >= len(buckets) {
			activeSeparatorBucket = len(buckets) - 1
		}
	}

	for i, bucket := range buckets {
		if bucket.ItemCount == 0 {
			continue
		}

		row := displayStartRow + i

		// Draw separator line between active and eliminated sections (only in MinimapShowAll mode)
		if r.cfg.MinimapShowAll && i == activeSeparatorBucket && len(eliminatedItems) > 0 {
			separatorStyle := tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
			// Clear the row first
			for x := 0; x < width; x++ {
				r.screen.SetContent(startX + x, row, ' ', nil, tcell.StyleDefault)
			}
			// Draw the separator line
			for x := 0; x < width; x++ {
				r.screen.SetContent(startX + x, row, '─', nil, separatorStyle)
			}
			row++ // Move to next row for the actual bucket
			if row >= displayStartRow + actualDisplayHeight {
				break // Don't go beyond display area
			}
		}

		// Calculate averages
		bucket.AvgRank /= float64(bucket.ItemCount)
		bucket.AvgStdDev /= float64(bucket.ItemCount)

		// Determine color based on elimination status
		activeRatio := float64(activeCounts[i]) / float64(bucket.ItemCount)
		var barStyle tcell.Style

		if bucket.ContainsElbow {
			// Mark this row for cutoff line rendering, but render data first
			// (We'll render the cutoff line later to ensure it overwrites data)
			barStyle = tcell.StyleDefault.Foreground(tcell.ColorWhite) // Render normally first
		} else if activeRatio == 0 {
			// All items eliminated - use gray
			barStyle = tcell.StyleDefault.Foreground(tcell.ColorDarkGray)
		} else {
			// Active items - use white
			barStyle = tcell.StyleDefault.Foreground(tcell.ColorWhite)
		}

		// Calculate bar width (inverse of rank - better items get longer bars)
		// Normalize to 0-1 range based on min/max ranks
		var normalizedRank float64
		if maxRank > minRank {
			normalizedRank = (bucket.AvgRank - minRank) / (maxRank - minRank)
		}
		barWidth := int((1.0 - normalizedRank) * float64(maxBarWidth))

		if barWidth < 1 && bucket.ItemCount > 0 {
			barWidth = 1  // Always show at least one character
		}

		// Draw the bar using block characters for density
		for x := 0; x < barWidth && x < maxBarWidth; x++ {
			var ch rune
			if bucket.ContainsElbow {
				ch = '▓'  // Different character for elbow
			} else if x == barWidth-1 && barWidth < maxBarWidth {
				// Use partial blocks for the last character for smoother gradients
				remainder := ((1.0 - normalizedRank) * float64(maxBarWidth)) - float64(barWidth-1)
				if remainder < 0.25 {
					ch = '▎'
				} else if remainder < 0.5 {
					ch = '▌'
				} else if remainder < 0.75 {
					ch = '▊'
				} else {
					ch = '█'
				}
			} else {
				ch = '█'
			}
			r.screen.SetContent(startX + 1 + x, row, ch, nil, barStyle)
		}

		// Elbow position tracking (no visual marker needed)
	}

	// Draw current cutoff line (elbow) after all data bars are rendered
	for i, bucket := range buckets {
		if bucket.ContainsElbow {
			row := displayStartRow + i
			cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
			// Clear the row and draw cutoff line
			for x := 0; x < width; x++ {
				r.screen.SetContent(startX + x, row, ' ', nil, tcell.StyleDefault)
			}
			for x := 0; x < width; x++ {
				r.screen.SetContent(startX + x, row, '─', nil, cutoffStyle)
			}
			break // Only one elbow per round
		}
	}

	// Draw historical cutoff lines from previous rounds (only in MinimapShowAll mode)
	if r.cfg.MinimapShowAll && len(r.roundCutoffs) > 0 {
		for _, cutoff := range r.roundCutoffs {
			// The cutoff position needs to be mapped to the new sorted order (active first, then eliminated)
			// Find where this cutoff falls in our new active/eliminated arrangement
			var cutoffIndex int
			if cutoff.Position <= len(activeItems) {
				// Cutoff is within active items
				cutoffIndex = cutoff.Position
			} else {
				// Cutoff includes some eliminated items, map to boundary between active and eliminated
				cutoffIndex = len(activeItems)
			}

			// Convert to minimap bucket
			cutoffBucket := int(float64(cutoffIndex) / itemsPerRow)
			if cutoffBucket >= len(buckets) {
				cutoffBucket = len(buckets) - 1
			}

			cutoffRow := displayStartRow + cutoffBucket
			if cutoffRow >= displayStartRow && cutoffRow < displayStartRow + actualDisplayHeight {
				// Use gray for historical cutoff lines (matches eliminated items)
				cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorDarkGray).Bold(true)

				// Clear the row first to avoid overlapping with data bars
				for x := 0; x < width; x++ {
					r.screen.SetContent(startX + x, cutoffRow, ' ', nil, tcell.StyleDefault)
				}

				// Draw horizontal cutoff line
				for x := 0; x < width; x++ {
					r.screen.SetContent(startX + x, cutoffRow, '─', nil, cutoffStyle)
				}
			}
		}
	}

	// Scale indicator removed for cleaner minimap
}

func (r *Ranker) renderVisualization(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if r.screen == nil {
		return
	}

	r.screen.Clear()
	width, height := r.screen.Size()

	// Update allRoundStats for minimap support (even in normal mode)
	// In normal mode, itemStatsMap only contains active items for this round,
	// so we only update active items, preserving eliminated items' frozen stats
	if r.cfg.MinimapEnabled && r.allRoundStats != nil {
		for id, stats := range itemStatsMap {
			r.allRoundStats[id] = stats
		}
	}

	// Sort items by appropriate metric
	var sortedItems []*itemStats
	var displayMode string
	if r.cfg.StdDevElbow {
		// Sort by std dev (most stable first)
		sortedItems = sortByStdDev(itemStatsMap)
		displayMode = "STABILITY MODE"
	} else {
		// Sort by average rank (existing)
		sortedItems = sortByAvgRank(itemStatsMap)
		displayMode = "RANK MODE"
	}

	// Determine cutoff for visualization (show where the cut would be)
	cutoffIndex := r.determineCutoff(sortedItems, round)

	// Calculate detailed cutoff information for debugging
	threshold := r.getThreshold(round)
	topBlockSize := 0
	for _, stats := range sortedItems {
		if stats.stdDev <= threshold {
			topBlockSize++
		} else {
			break
		}
	}

	// Calculate bottom block
	bottomBlockSize := 0
	for i := len(sortedItems) - 1; i >= 0; i-- {
		if sortedItems[i].stdDev <= threshold {
			bottomBlockSize++
		} else {
			break
		}
	}

	analysisEnd := len(sortedItems) - bottomBlockSize



	// Calculate current elbow index for display
	var scores []float64
	for _, stats := range sortedItems {
		scores = append(scores, stats.avgRank)
	}
	currentElbowIndex := detectElbow(scores)

	// Calculate current elbow range for display
	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	rangeThreshold := r.cfg.CutoffRangePercent * float64(len(sortedItems))

	// Check for no-signal condition
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	// Determine cutoff method for display
	var cutoffMethod string
	if r.consecutiveStableCount >= r.cfg.StableRuns {
		cutoffMethod = "MAX(stable)"
	} else {
		cutoffMethod = "MEDIAN(unstable)"
	}

	// Enhanced header with debug info
	var headerStr string
	if noSignalDetected {
		headerStr = fmt.Sprintf("%s | Round: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d | NO SIGNAL - NOISE DETECTED",
			displayMode, round, iteration, maxIterations,
			len(sortedItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	} else {
		headerStr = fmt.Sprintf("%s | Round: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d",
			displayMode, round, iteration, maxIterations,
			len(sortedItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	}
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorWhite))


	// Add explanation when cutoff equals total items (only after min iterations)
	if cutoffIndex >= len(sortedItems) && iteration >= minIterations {
		warningStr := fmt.Sprintf("WARNING: Cutoff=%d equals total items=%d - will select all, triggering stop condition",
			cutoffIndex, len(sortedItems))
		r.writeString(0, 2, warningStr, tcell.StyleDefault.Foreground(tcell.ColorRed))
	}

	// Render help text
	helpStr := "Press Ctrl+C, Esc, or 'q' to quit"
	helpRow := 2
	if cutoffIndex >= len(sortedItems) {
		helpRow = 3 // Move help text down if warning is shown
	}
	r.writeString(0, helpRow, helpStr, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))

	// Render each item
	startRow := helpRow + 1
	for i, item := range sortedItems {
		if i >= height-(startRow+1) {
			break // Don't render beyond screen height (header + debug + help + margin)
		}

		row := i + startRow // Offset for header, debug, and help text

		// Determine color based on elbow position and convergence
		isAboveElbow := i < currentElbowIndex
		isConverged := item.stdDev <= threshold

		var style tcell.Style
		if isAboveElbow && isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)  // Above elbow and converged
		} else if isAboveElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorYellow) // Above elbow but not converged
		} else if isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorBlue)   // Below elbow but converged
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorWhite)  // Below elbow and not converged
		}

		// Calculate bar lengths
		stdDevBarLen := int(math.Min(float64(item.stdDev*10), 100))
		scoreBarLen := int(math.Max(0, 100-item.avgRank*10))

		centerX := width / 2

		// Render std dev value (left of std dev bar)
		stdDevStr := fmt.Sprintf("%.1f", item.stdDev)
		// Position std dev value to the left of the new bar start position
		barStartX := centerX - len(item.ID)/2 - 2
		stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
		if stdDevX < 0 {
			stdDevX = 0
		}
		r.writeString(stdDevX, row, stdDevStr, style)

		// Render std dev bar (growing left from center, with space buffer)
		for j := 0; j < stdDevBarLen && centerX-len(item.ID)/2-j-2 >= 0; j++ {
			r.screen.SetContent(centerX-len(item.ID)/2-j-2, row, '=', nil, style)
		}

		// Render item ID at center
		idX := centerX - len(item.ID)/2
		if idX < 0 {
			idX = 0
		}
		r.writeString(idX, row, item.ID, style)

		// Render score bar with actual item text
		startX := centerX + len(item.ID)/2 + 1
		availableWidth := scoreBarLen
		if startX+availableWidth > width {
			availableWidth = width - startX
		}

		if availableWidth > 0 {
			// Use the item's actual text, truncated to fit
			displayText := item.Value
			if len(displayText) > availableWidth {
				displayText = displayText[:availableWidth]
			}

			// Render the text
			for i, ch := range displayText {
				if startX+i < width {
					r.screen.SetContent(startX+i, row, ch, nil, style)
				}
			}

			// Fill remaining space with + characters
			for j := len(displayText); j < availableWidth && startX+j < width; j++ {
				r.screen.SetContent(startX+j, row, '=', nil, style)
			}
		}

		// Render rank value (right of score bar)
		rankStr := fmt.Sprintf("%.1f", item.avgRank)
		rankX := startX + scoreBarLen + 1
		if rankX+len(rankStr) > width {
			rankX = width - len(rankStr)
		}
		if rankX >= 0 {
			r.writeString(rankX, row, rankStr, style)
		}
	}

	// Draw live cutoff line to show where the cut will be
	if cutoffIndex > 0 && cutoffIndex < len(sortedItems) {
		cutoffRow := cutoffIndex + 2 // Offset for header lines
		if cutoffRow < height-1 {
			// Draw horizontal line across screen
			cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
			if r.cfg.InvertCutoff {
				cutoffStyle = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
			}
			for x := 0; x < width; x++ {
				r.screen.SetContent(x, cutoffRow, '─', nil, cutoffStyle)
			}

			// Add cutoff message
			var cutoffMsg string
			if r.cfg.InvertCutoff {
				cutoffMsg = fmt.Sprintf(" INVERTED CUTOFF: Bottom %d would advance ", len(sortedItems)-cutoffIndex)
			} else {
				cutoffMsg = fmt.Sprintf(" CUTOFF: Top %d would advance ", cutoffIndex)
			}
			msgX := (width - len(cutoffMsg)) / 2
			if msgX >= 0 {
				r.writeString(msgX, cutoffRow, cutoffMsg, cutoffStyle)
			}
		}
	} else {
		// Explain why no cutoff line
		explanationRow := height / 2
		var explanation string
		var explanationStyle tcell.Style

		if cutoffIndex <= 0 {
			explanation = "NO CUTOFF LINE: Cutoff at position 0 (nothing selected)"
			explanationStyle = tcell.StyleDefault.Foreground(tcell.ColorRed)
		} else if cutoffIndex >= len(sortedItems) {
			explanation = fmt.Sprintf("NO CUTOFF LINE: Cutoff=%d includes all %d items (stop condition)",
				cutoffIndex, len(sortedItems))
			explanationStyle = tcell.StyleDefault.Foreground(tcell.ColorYellow)

			// Add more detail about why
			if topBlockSize == len(sortedItems) {
				explanation += " - All items converged into top block"
			} else if analysisEnd <= topBlockSize {
				explanation += fmt.Sprintf(" - Bottom block (%d) meets top block (%d)",
					bottomBlockSize, topBlockSize)
			}
		}

		if explanation != "" && explanationRow < height-1 {
			// Clear and draw explanation
			for x := 0; x < width; x++ {
				r.screen.SetContent(x, explanationRow, '─', nil, explanationStyle)
			}
			msgX := (width - len(explanation)) / 2
			if msgX >= 0 && msgX+len(explanation) <= width {
				r.writeString(msgX, explanationRow, explanation, explanationStyle.Bold(true))
			}
		}
	}

	r.screen.Show()
}

func (r *Ranker) writeString(x, y int, s string, style tcell.Style) {
	for i, ch := range s {
		r.screen.SetContent(x+i, y, ch, nil, style)
	}
}

// renderCompletedRound function removed - now just pause on last iteration frame





func (r *Ranker) renderFinalVisualization(itemStatsMap map[string]*itemStats, round int) {
	if r.screen == nil {
		return
	}

	r.screen.Clear()
	width, height := r.screen.Size()

	// Calculate stats for final display
	avgStdDev := calculateAvgStdDev(itemStatsMap)

	// Sort items by appropriate metric
	var sortedItems []*itemStats
	var displayMode string
	if r.cfg.StdDevElbow {
		// Sort by std dev (most stable first)
		sortedItems = sortByStdDev(itemStatsMap)
		displayMode = "STABILITY MODE"
	} else {
		// Sort by average rank (existing)
		sortedItems = sortByAvgRank(itemStatsMap)
		displayMode = "RANK MODE"
	}

	// Determine threshold for this round
	threshold := r.getThreshold(round)



	// Calculate current elbow index for display
	var scores []float64
	for _, stats := range sortedItems {
		scores = append(scores, stats.avgRank)
	}
	currentElbowIndex := detectElbow(scores)

	// Render header showing round completion
	roundName := "Threshing"
	if round > 1 {
		roundName = "Ranking"
	}
	// Calculate current elbow range for display
	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	rangeThreshold := r.cfg.CutoffRangePercent * float64(len(sortedItems))

	// Check for no-signal condition
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	var headerStr string
	if noSignalDetected {
		headerStr = fmt.Sprintf("%s | Round %d (%s) FINAL | Avg Std Dev: %.2f | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d | NO SIGNAL - NOISE DETECTED",
			displayMode, round, roundName, avgStdDev, currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	} else {
		headerStr = fmt.Sprintf("%s | Round %d (%s) FINAL | Avg Std Dev: %.2f | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d",
			displayMode, round, roundName, avgStdDev, currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	}
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorGreen))


	// Render each item (same as regular visualization)
	for i, item := range sortedItems {
		if i >= height-4 {
			break
		}

		row := i + 3

		// Determine color based on elbow position and convergence
		isAboveElbow := i < currentElbowIndex
		isConverged := item.stdDev <= threshold

		var style tcell.Style
		if isAboveElbow && isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)  // Above elbow and converged
		} else if isAboveElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorYellow) // Above elbow but not converged
		} else if isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorBlue)   // Below elbow but converged
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorWhite)  // Below elbow and not converged
		}

		// Calculate bar lengths
		stdDevBarLen := int(math.Min(float64(item.stdDev*10), 100))
		scoreBarLen := int(math.Max(0, 100-item.avgRank*10))

		centerX := width / 2

		// Render std dev value
		stdDevStr := fmt.Sprintf("%.1f", item.stdDev)
		// Position std dev value to the left of the new bar start position
		barStartX := centerX - len(item.ID)/2 - 2
		stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
		if stdDevX < 0 {
			stdDevX = 0
		}
		r.writeString(stdDevX, row, stdDevStr, style)

		// Render std dev bar (growing left from center, with space buffer)
		for j := 0; j < stdDevBarLen && centerX-len(item.ID)/2-j-2 >= 0; j++ {
			r.screen.SetContent(centerX-len(item.ID)/2-j-2, row, '=', nil, style)
		}

		// Render item ID at center
		idX := centerX - len(item.ID)/2
		if idX < 0 {
			idX = 0
		}
		r.writeString(idX, row, item.ID, style)

		// Render score bar with actual item text
		startX := centerX + len(item.ID)/2 + 1
		availableWidth := scoreBarLen
		if startX+availableWidth > width {
			availableWidth = width - startX
		}

		if availableWidth > 0 {
			// Use the item's actual text, truncated to fit
			displayText := item.Value
			if len(displayText) > availableWidth {
				displayText = displayText[:availableWidth]
			}

			// Render the text
			for i, ch := range displayText {
				if startX+i < width {
					r.screen.SetContent(startX+i, row, ch, nil, style)
				}
			}

			// Fill remaining space with + characters
			for j := len(displayText); j < availableWidth && startX+j < width; j++ {
				r.screen.SetContent(startX+j, row, '=', nil, style)
			}
		}

		// Render rank value
		rankStr := fmt.Sprintf("%.1f", item.avgRank)
		rankX := startX + scoreBarLen + 1
		if rankX+len(rankStr) > width {
			rankX = width - len(rankStr)
		}
		if rankX >= 0 {
			r.writeString(rankX, row, rankStr, style)
		}
	}

	r.screen.Show()
}

func (r *Ranker) pauseWithVisualization(round int, totalSeconds int) {
	if r.screen == nil {
		return
	}

	r.cfg.Logger.Info("Starting visualization pause", "round", round, "seconds", totalSeconds)

	// Keep the visualization on screen and just update the pause info
	for remaining := totalSeconds; remaining > 0; remaining-- {
		// Update just the pause info line (line 1)
		pauseStr := fmt.Sprintf("Pausing for %d seconds - Press any key to continue", remaining)
		// Clear the line first
		for i := 0; i < 100; i++ { // Clear a wider area
			r.screen.SetContent(i, 1, ' ', nil, tcell.StyleDefault)
		}
		// Write updated pause info
		r.writeString(0, 1, pauseStr, tcell.StyleDefault.Foreground(tcell.ColorYellow))

		r.screen.Show()
		time.Sleep(1 * time.Second)
	}

	r.cfg.Logger.Info("Visualization pause completed", "round", round)
}

func (r *Ranker) renderFinalStop(reason StopReason, round int) {
	if r.screen == nil {
		return
	}

	// DON'T clear the screen - preserve the debug information
	width, height := r.screen.Size()

	// Determine header color based on stop reason
	var headerStyle tcell.Style
	switch reason {
	case StopReasonSingleItem, StopReasonAllConverged:
		headerStyle = tcell.StyleDefault.Foreground(tcell.ColorGreen).Bold(true)
	case StopReasonNoSignal:
		headerStyle = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
	case StopReasonNoMoreCuts, StopReasonTooFewItems:
		headerStyle = tcell.StyleDefault.Foreground(tcell.ColorOrange).Bold(true)
	default:
		headerStyle = tcell.StyleDefault.Foreground(tcell.ColorWhite).Bold(true)
	}

	// Clear bottom area for stop information (preserve top debug info)
	bottomStartRow := height - 6
	for y := bottomStartRow; y < height; y++ {
		for x := 0; x < width; x++ {
			r.screen.SetContent(x, y, ' ', nil, tcell.StyleDefault)
		}
	}

	// Render stop header at bottom of screen
	stopHeader := fmt.Sprintf("RANKING STOPPED - Round %d", round)
	r.writeString((width-len(stopHeader))/2, bottomStartRow, stopHeader, headerStyle)

	// Render stop reason
	reasonStr := reason.String()
	r.writeString((width-len(reasonStr))/2, bottomStartRow+1, reasonStr, headerStyle)

	// Add explanatory text based on reason
	var explanation string
	switch reason {
	case StopReasonNoSignal:
		explanation = "The cutoff position stabilized but no items above it have converged."
	case StopReasonNoMoreCuts:
		explanation = "Cannot find a meaningful cutoff point - all items appear equivalent."
	case StopReasonTooFewItems:
		explanation = "Not enough items remain to continue refinement."
	case StopReasonSingleItem:
		explanation = "The ranking has converged to a single best item."
	}

	if explanation != "" {
		r.writeString((width-len(explanation))/2, bottomStartRow+2, explanation, tcell.StyleDefault.Foreground(tcell.ColorGray))
	}

	// Show pause info if configured
	if r.cfg.RoundPause > 0 {
		pauseStr := fmt.Sprintf("Final results - Pausing %d seconds - Press Ctrl+C, Esc, or 'q' to quit",
			r.cfg.RoundPause)
		r.writeString(0, bottomStartRow+4, pauseStr, tcell.StyleDefault.Foreground(tcell.ColorYellow))
	} else {
		// Show permanent message when no pause is configured
		permanentStr := "Final results displayed above - Press Ctrl+C, Esc, or 'q' to quit"
		r.writeString(0, bottomStartRow+4, permanentStr, tcell.StyleDefault.Foreground(tcell.ColorYellow))
	}

	r.screen.Show()
}

// Progressive visualization functions
func (r *Ranker) renderProgressiveVisualizationWithOptionalMinimap(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if !r.cfg.MinimapEnabled || !r.cfg.Observe || r.screen == nil {
		r.renderProgressiveVisualization(itemStatsMap, round, iteration, minIterations, maxIterations)
		return
	}

	r.renderProgressiveWithMinimap(itemStatsMap, round, iteration, minIterations, maxIterations)
}

func (r *Ranker) renderProgressiveWithMinimap(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if r.screen == nil {
		return
	}

	r.screen.Clear()
	width, height := r.screen.Size()

	// Calculate layout dimensions
	mainWidth := int(float64(width) * 0.8)
	minimapStartX := mainWidth + 1
	minimapWidth := width - minimapStartX - 1

	// Must have at least 5 chars for minimap to be useful
	if minimapWidth < 5 {
		// Fall back to full-width display
		r.renderProgressiveVisualization(itemStatsMap, round, iteration, minIterations, maxIterations)
		return
	}

	// Draw vertical separator
	separatorStyle := tcell.StyleDefault.Foreground(tcell.ColorDarkGray)
	for y := 0; y < height; y++ {
		r.screen.SetContent(mainWidth, y, '│', nil, separatorStyle)
	}

	// Render main progressive display (constrained to left portion)
	r.renderProgressiveDisplayConstrained(itemStatsMap, round, iteration, minIterations, maxIterations, mainWidth, height)

	// Render minimap on right
	r.renderMinimap(itemStatsMap, minimapStartX, minimapWidth, height, round)

	r.screen.Show()
}

func (r *Ranker) renderProgressiveDisplayConstrained(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int, maxWidth int, maxHeight int) {
	// Initialize display states on first call
	if len(r.displayStates) == 0 {
		for id, stats := range itemStatsMap {
			r.displayStates[id] = &ItemDisplayState{
				ID:              id,
				Value:           stats.Value,
				RoundEliminated: 0,
				LastRank:        stats.avgRank,
				LastStdDev:      stats.stdDev,
			}
			// Keep a copy of all stats
			r.allRoundStats[id] = stats
		}
	}

	// Update ONLY active items (don't update eliminated items)
	for id, stats := range itemStatsMap {
		if state, exists := r.displayStates[id]; exists {
			// Only update if item is still active (not eliminated)
			if state.RoundEliminated == 0 {
				state.LastRank = stats.avgRank
				state.LastStdDev = stats.stdDev
				r.allRoundStats[id] = stats
			}
			// If eliminated, keep the frozen stats from elimination time
		}
	}

	// Sort items by appropriate metric (same as normal viz)
	var activeItems []*itemStats
	var displayMode string
	if r.cfg.StdDevElbow {
		// Sort by std dev (most stable first)
		activeItems = sortByStdDev(itemStatsMap)
		displayMode = "STABILITY MODE"
	} else {
		// Sort by average rank (existing)
		activeItems = sortByAvgRank(itemStatsMap)
		displayMode = "RANK MODE"
	}

	// Determine cutoff for visualization
	cutoffIndex := r.determineCutoff(activeItems, round)

	// Calculate debug info (same as normal viz)
	threshold := r.getThreshold(round)
	topBlockSize := 0
	for _, stats := range activeItems {
		if stats.stdDev <= threshold {
			topBlockSize++
		} else {
			break
		}
	}

	bottomBlockSize := 0
	if topBlockSize < len(activeItems) {
		for i := len(activeItems) - 1; i >= topBlockSize; i-- {
			if activeItems[i].stdDev <= threshold {
				bottomBlockSize++
			} else {
				break
			}
		}
	}
	if bottomBlockSize > len(activeItems) - topBlockSize {
		bottomBlockSize = len(activeItems) - topBlockSize
	}

	analysisEnd := len(activeItems) - bottomBlockSize

	// Calculate current elbow index for display
	var scores []float64
	for _, stats := range activeItems {
		scores = append(scores, stats.avgRank)
	}
	currentElbowIndex := detectElbow(scores)

	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	rangeThreshold := r.cfg.CutoffRangePercent * float64(len(activeItems))

	// Check for no-signal condition
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	// Determine cutoff method for display
	var cutoffMethod string
	if r.consecutiveStableCount >= r.cfg.StableRuns {
		cutoffMethod = "MAX(stable)"
	} else {
		cutoffMethod = "MEDIAN(unstable)"
	}

	// Enhanced header with debug info (same as normal viz)
	var headerStr string
	if noSignalDetected {
		headerStr = fmt.Sprintf("PROGRESSIVE %s | Round: %d | Active: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d | NO SIGNAL - NOISE DETECTED",
			displayMode, round, len(itemStatsMap), iteration, maxIterations,
			len(activeItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	} else {
		headerStr = fmt.Sprintf("PROGRESSIVE %s | Round: %d | Active: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d",
			displayMode, round, len(itemStatsMap), iteration, maxIterations,
			len(activeItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	}

	// Truncate header if needed
	if len(headerStr) > maxWidth {
		headerStr = headerStr[:maxWidth-3] + "..."
	}
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorWhite))

	// Add explanation when cutoff equals total items (only after min iterations)
	startRow := 2
	if cutoffIndex >= len(activeItems) && iteration >= minIterations {
		warningStr := fmt.Sprintf("WARNING: Cutoff=%d equals total items=%d - will select all, triggering stop condition",
			cutoffIndex, len(activeItems))
		if len(warningStr) > maxWidth {
			warningStr = warningStr[:maxWidth-3] + "..."
		}
		r.writeString(0, 2, warningStr, tcell.StyleDefault.Foreground(tcell.ColorRed))
		startRow = 3
	}

	// Build complete progressive view with cutoffs at exact positions
	r.renderProgressiveItemsWithCutoffsConstrained(startRow, activeItems, round, maxWidth, maxHeight, threshold, currentElbowIndex, cutoffIndex)
}

func (r *Ranker) renderProgressiveVisualization(itemStatsMap map[string]*itemStats, round int, iteration int, minIterations int, maxIterations int) {
	if r.screen == nil {
		return
	}

	r.screen.Clear()
	width, height := r.screen.Size()

	// Initialize display states on first call
	if len(r.displayStates) == 0 {
		for id, stats := range itemStatsMap {
			r.displayStates[id] = &ItemDisplayState{
				ID:              id,
				Value:           stats.Value,
				RoundEliminated: 0,
				LastRank:        stats.avgRank,
				LastStdDev:      stats.stdDev,
			}
			// Keep a copy of all stats
			r.allRoundStats[id] = stats
		}
	}

	// Update ONLY active items (don't update eliminated items)
	for id, stats := range itemStatsMap {
		if state, exists := r.displayStates[id]; exists {
			// Only update if item is still active (not eliminated)
			if state.RoundEliminated == 0 {
				state.LastRank = stats.avgRank
				state.LastStdDev = stats.stdDev
				r.allRoundStats[id] = stats
			}
			// If eliminated, keep the frozen stats from elimination time
		}
	}

	// Sort items by appropriate metric (same as normal viz)
	var activeItems []*itemStats
	var displayMode string
	if r.cfg.StdDevElbow {
		// Sort by std dev (most stable first)
		activeItems = sortByStdDev(itemStatsMap)
		displayMode = "STABILITY MODE"
	} else {
		// Sort by average rank (existing)
		activeItems = sortByAvgRank(itemStatsMap)
		displayMode = "RANK MODE"
	}

	// Determine cutoff for visualization
	cutoffIndex := r.determineCutoff(activeItems, round)

	// Calculate debug info (same as normal viz)
	threshold := r.getThreshold(round)
	topBlockSize := 0
	for _, stats := range activeItems {
		if stats.stdDev <= threshold {
			topBlockSize++
		} else {
			break
		}
	}

	bottomBlockSize := 0
	if topBlockSize < len(activeItems) {
		for i := len(activeItems) - 1; i >= topBlockSize; i-- {
			if activeItems[i].stdDev <= threshold {
				bottomBlockSize++
			} else {
				break
			}
		}
	}
	if bottomBlockSize > len(activeItems) - topBlockSize {
		bottomBlockSize = len(activeItems) - topBlockSize
	}

	analysisEnd := len(activeItems) - bottomBlockSize

	// Calculate current elbow index for display
	var scores []float64
	for _, stats := range activeItems {
		scores = append(scores, stats.avgRank)
	}
	currentElbowIndex := detectElbow(scores)

	elbowRange := 0
	if len(r.recentElbowPositions) >= 2 {
		elbowRange = maxInt(r.recentElbowPositions) - minInt(r.recentElbowPositions)
	}

	rangeThreshold := r.cfg.CutoffRangePercent * float64(len(activeItems))

	// Check for no-signal condition
	noSignalDetected := false
	for _, stats := range itemStatsMap {
		if stats.noSignal {
			noSignalDetected = true
			break
		}
	}

	// Determine cutoff method for display
	var cutoffMethod string
	if r.consecutiveStableCount >= r.cfg.StableRuns {
		cutoffMethod = "MAX(stable)"
	} else {
		cutoffMethod = "MEDIAN(unstable)"
	}

	// Enhanced header with debug info (same as normal viz)
	var headerStr string
	if noSignalDetected {
		headerStr = fmt.Sprintf("PROGRESSIVE %s | Round: %d | Active: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d | NO SIGNAL - NOISE DETECTED",
			displayMode, round, len(itemStatsMap), iteration, maxIterations,
			len(activeItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	} else {
		headerStr = fmt.Sprintf("PROGRESSIVE %s | Round: %d | Active: %d | Completed: %d/%d | Items: %d | TopBlock: %d | BottomBlock: %d | AnalysisRange: %d | Cutoff: %d (%s) | Elbow: %d (Range=%d/%.1f) | Stable: %d/%d",
			displayMode, round, len(itemStatsMap), iteration, maxIterations,
			len(activeItems), topBlockSize, bottomBlockSize, analysisEnd, cutoffIndex, cutoffMethod,
			currentElbowIndex, elbowRange, rangeThreshold, r.consecutiveStableCount, r.cfg.StableRuns)
	}
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorWhite))


	// Add explanation when cutoff equals total items (only after min iterations)
	startRow := 2
	if cutoffIndex >= len(activeItems) && iteration >= minIterations {
		warningStr := fmt.Sprintf("WARNING: Cutoff=%d equals total items=%d - will select all, triggering stop condition",
			cutoffIndex, len(activeItems))
		r.writeString(0, 2, warningStr, tcell.StyleDefault.Foreground(tcell.ColorRed))
		startRow = 3
	}

	// Render active items (same as normal viz)
	activeRowsNeeded := len(activeItems)
	if activeRowsNeeded > height - startRow - 5 { // Leave room for eliminated items
		activeRowsNeeded = height - startRow - 5
	}

	for i, item := range activeItems {
		if i >= activeRowsNeeded {
			break
		}

		row := i + startRow

		// Use same styling as normal viz
		isAboveElbow := i < currentElbowIndex
		isConverged := item.stdDev <= threshold

		var style tcell.Style
		if isAboveElbow && isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
		} else if isAboveElbow {
			style = tcell.StyleDefault.Foreground(tcell.ColorYellow)
		} else if isConverged {
			style = tcell.StyleDefault.Foreground(tcell.ColorBlue)
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorWhite)
		}

		r.renderNormalItem(item, row, style, width)
	}

	// Build complete progressive view with cutoffs at exact positions
	// Note: Current cutoff line will be drawn within the progressive rendering
	r.renderProgressiveItemsWithCutoffs(startRow, activeItems, round, width, height, threshold, currentElbowIndex, cutoffIndex)

	r.screen.Show()
}

func (r *Ranker) renderProgressiveItemsWithCutoffsConstrained(startRow int, activeItems []*itemStats, currentRound int, maxWidth int, maxHeight int, threshold float64, currentElbowIndex int, currentCutoffIndex int) {
	// For now, just call the existing function with width constraint
	// This could be optimized further to handle width constraints in the progressive rendering itself
	r.renderProgressiveItemsWithCutoffs(startRow, activeItems, currentRound, maxWidth, maxHeight, threshold, currentElbowIndex, currentCutoffIndex)
}

func (r *Ranker) renderProgressiveItemsWithCutoffs(startRow int, activeItems []*itemStats, currentRound int, width int, height int, threshold float64, currentElbowIndex int, currentCutoffIndex int) {
	// Build a complete list of all items with their elimination info
	type DisplayItem struct {
		Stats           *itemStats
		State           *ItemDisplayState
		IsActive        bool
		EliminatedRound int
		Position        int // Original position in the dataset for this round
	}

	var allItems []DisplayItem

	// Add active items first
	for i, stats := range activeItems {
		state := r.displayStates[stats.ID]
		allItems = append(allItems, DisplayItem{
			Stats:           stats,
			State:           state,
			IsActive:        true,
			EliminatedRound: 0,
			Position:        i,
		})
	}

	// Add eliminated items, preserving their positions from when they were eliminated
	for _, cutoff := range r.roundCutoffs {
		// Find items eliminated in this round
		var eliminatedInThisRound []DisplayItem
		for _, state := range r.displayStates {
			if state.RoundEliminated == cutoff.Round {
				eliminatedStats := r.allRoundStats[state.ID]
				eliminatedInThisRound = append(eliminatedInThisRound, DisplayItem{
					Stats:           eliminatedStats,
					State:           state,
					IsActive:        false,
					EliminatedRound: cutoff.Round,
				})
			}
		}

		// Sort eliminated items by their frozen rank at elimination time
		sort.Slice(eliminatedInThisRound, func(i, j int) bool {
			if eliminatedInThisRound[i].Stats.avgRank == eliminatedInThisRound[j].Stats.avgRank {
				return eliminatedInThisRound[i].Stats.ID < eliminatedInThisRound[j].Stats.ID
			}
			return eliminatedInThisRound[i].Stats.avgRank < eliminatedInThisRound[j].Stats.avgRank
		})

		// Insert eliminated items at the cutoff position
		insertPos := cutoff.Position
		if insertPos > len(allItems) {
			insertPos = len(allItems)
		}

		// Insert cutoff marker first
		allItems = append(allItems[:insertPos], append([]DisplayItem{{
			Stats:           nil, // Marker for cutoff line
			State:           nil,
			IsActive:        false,
			EliminatedRound: cutoff.Round,
			Position:        -1, // Special marker
		}}, append(eliminatedInThisRound, allItems[insertPos:]...)...)...)
	}

	// Render the unified list
	row := startRow
	activeItemsRendered := 0

	for _, item := range allItems {
		if row >= height-1 {
			break
		}

		// Check if we need to draw the current cutoff line before this item
		if item.IsActive && activeItemsRendered == currentCutoffIndex && currentCutoffIndex > 0 && currentCutoffIndex < len(activeItems) {
			// Clear the entire row first to prevent text bleed-through
			for x := 0; x < width; x++ {
				r.screen.SetContent(x, row, ' ', nil, tcell.StyleDefault)
			}

			// Draw current cutoff line (limited to 100 chars, centered)
			cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorRed).Bold(true)
			if r.cfg.InvertCutoff {
				cutoffStyle = tcell.StyleDefault.Foreground(tcell.ColorYellow).Bold(true)
			}
			lineWidth := width
			if lineWidth > 100 {
				lineWidth = 100
			}
			// Center the line in the terminal
			lineStartX := (width - lineWidth) / 2
			for x := 0; x < lineWidth; x++ {
				r.screen.SetContent(lineStartX+x, row, '─', nil, cutoffStyle)
			}

			var cutoffMsg string
			if r.cfg.InvertCutoff {
				cutoffMsg = fmt.Sprintf(" CURRENT CUTOFF: Bottom %d advance to next round ", len(activeItems)-currentCutoffIndex)
			} else {
				cutoffMsg = fmt.Sprintf(" CURRENT CUTOFF: Top %d advance to next round ", currentCutoffIndex)
			}
			msgX := lineStartX + (lineWidth-len(cutoffMsg))/2
			if msgX >= 0 && msgX+len(cutoffMsg) <= width {
				r.writeString(msgX, row, cutoffMsg, cutoffStyle)
			}
			row++
			if row >= height-1 {
				break
			}
		}

		// Check if this is a historical cutoff line marker
		if item.Position == -1 && item.Stats == nil {
			// Clear the entire row first to prevent text bleed-through
			for x := 0; x < width; x++ {
				r.screen.SetContent(x, row, ' ', nil, tcell.StyleDefault)
			}

			// Draw historical cutoff line (limited to 100 chars, centered)
			cutoffStyle := tcell.StyleDefault.Foreground(tcell.ColorGray).Bold(true)
			lineWidth := width
			if lineWidth > 100 {
				lineWidth = 100
			}
			// Center the line in the terminal
			lineStartX := (width - lineWidth) / 2
			for x := 0; x < lineWidth; x++ {
				r.screen.SetContent(lineStartX+x, row, '─', nil, cutoffStyle)
			}
			cutoffMsg := fmt.Sprintf(" CUTOFF FROM ROUND %d ", item.EliminatedRound)
			msgX := lineStartX + (lineWidth-len(cutoffMsg))/2
			if msgX >= 0 && msgX+len(cutoffMsg) <= width {
				r.writeString(msgX, row, cutoffMsg, cutoffStyle)
			}
			row++
			continue
		}

		// Render item on current row
		if item.IsActive {
			// Active item - use normal styling
			isAboveElbow := activeItemsRendered < currentElbowIndex
			isConverged := item.Stats.stdDev <= threshold

			var style tcell.Style
			if isAboveElbow && isConverged {
				style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
			} else if isAboveElbow {
				style = tcell.StyleDefault.Foreground(tcell.ColorYellow)
			} else if isConverged {
				style = tcell.StyleDefault.Foreground(tcell.ColorBlue)
			} else {
				style = tcell.StyleDefault.Foreground(tcell.ColorWhite)
			}
			r.renderNormalItem(item.Stats, row, style, width)
			activeItemsRendered++
		} else {
			// Eliminated item - use gray styling
			roundsSinceElimination := currentRound - item.EliminatedRound
			grayLevel := 160 - (roundsSinceElimination * 20)
			if grayLevel < 80 {
				grayLevel = 80
			}
			color := tcell.NewRGBColor(int32(grayLevel), int32(grayLevel), int32(grayLevel))
			style := tcell.StyleDefault.Foreground(color)
			r.renderEliminatedItem(item.State, item.Stats, row, style, width)
		}
		row++ // Move to next row after rendering the item
	}
}

func (r *Ranker) drawGrayCutoffLine(row int, cutoff RoundCutoff, width int) {
	style := tcell.StyleDefault.Foreground(tcell.ColorGray)

	// Draw line
	for x := 0; x < width; x++ {
		r.screen.SetContent(x, row, '─', nil, style)
	}

	// Add label in center
	label := fmt.Sprintf(" Round %d: %d advance (%s) ",
		cutoff.Round, cutoff.ItemsBefore, cutoff.Method)
	labelX := (width - len(label)) / 2
	if labelX >= 0 && labelX+len(label) <= width {
		r.writeString(labelX, row, label, style)
	}
}

func (r *Ranker) renderProgressiveItem(item *ItemDisplayState, stats *itemStats, row int, isActive bool, currentRound int, width int) {
	// Determine color
	var style tcell.Style
	threshold := r.getThreshold(currentRound)

	if isActive {
		// Active items: use normal coloring
		if stats.stdDev <= threshold {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorYellow)
		}
	} else {
		// Frozen items: gray, darker for older eliminations
		roundsSinceElimination := currentRound - item.RoundEliminated
		grayLevel := 180 - (roundsSinceElimination * 30)
		if grayLevel < 80 {
			grayLevel = 80
		}
		color := tcell.NewRGBColor(int32(grayLevel), int32(grayLevel), int32(grayLevel))
		style = tcell.StyleDefault.Foreground(color)
	}

	centerX := width / 2

	// Render std dev bar and value
	stdDevBarLen := int(math.Min(float64(stats.stdDev*10), 100))
	stdDevStr := fmt.Sprintf("%.1f", stats.stdDev)
	barStartX := centerX - len(item.ID)/2 - 2
	stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
	if stdDevX < 0 {
		stdDevX = 0
	}
	r.writeString(stdDevX, row, stdDevStr, style)

	// Render std dev bar (growing left from center)
	for j := 0; j < stdDevBarLen && centerX-len(item.ID)/2-j-2 >= 0; j++ {
		r.screen.SetContent(centerX-len(item.ID)/2-j-2, row, '=', nil, style)
	}

	// Render item ID at center
	idX := centerX - len(item.ID)/2
	if idX < 0 {
		idX = 0
	}
	r.writeString(idX, row, item.ID, style)

	// Render score bar with item text
	startX := centerX + len(item.ID)/2 + 1
	scoreBarLen := int(math.Max(0, 100-stats.avgRank*10))
	availableWidth := scoreBarLen
	if startX+availableWidth > width {
		availableWidth = width - startX
	}

	if availableWidth > 0 {
		// Use the item's actual text, truncated to fit
		displayText := item.Value
		if len(displayText) > availableWidth {
			displayText = displayText[:availableWidth]
		}

		// Render the text
		for i, ch := range displayText {
			if startX+i < width {
				r.screen.SetContent(startX+i, row, ch, nil, style)
			}
		}

		// Fill remaining space with → characters
		for j := len(displayText); j < availableWidth && startX+j < width; j++ {
			r.screen.SetContent(startX+j, row, '=', nil, style)
		}
	}

	// Render rank value
	rankStr := fmt.Sprintf("%.1f", stats.avgRank)
	rankX := startX + scoreBarLen + 1
	if rankX+len(rankStr) > width {
		rankX = width - len(rankStr)
	}
	if rankX >= 0 {
		r.writeString(rankX, row, rankStr, style)
	}
}

func (r *Ranker) renderProgressiveHeader(round int, iteration int, minIterations int, maxIterations int, activeItems int) {
	// Progressive header
	headerStr := fmt.Sprintf("PROGRESSIVE | Round: %d | Active: %d/%d | Iteration: %d/%d",
		round, activeItems, len(r.displayStates), iteration, maxIterations)
	r.writeString(0, 0, headerStr, tcell.StyleDefault.Foreground(tcell.ColorWhite))

	// Help text
	helpStr := "Gray items eliminated in previous rounds | Press Ctrl+C, Esc, or 'q' to quit"
	r.writeString(0, 1, helpStr, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))

	// Add cutoff method info
	var cutoffMethod string
	if r.consecutiveStableCount >= r.cfg.StableRuns {
		cutoffMethod = "MAX(stable)"
	} else {
		cutoffMethod = "MEDIAN(unstable)"
	}
	cutoffStr := fmt.Sprintf("Current cutoff method: %s | Stable: %d/%d",
		cutoffMethod, r.consecutiveStableCount, r.cfg.StableRuns)
	r.writeString(0, 2, cutoffStr, tcell.StyleDefault.Foreground(tcell.ColorDarkGray))
}

func (r *Ranker) renderNormalItem(item *itemStats, row int, style tcell.Style, width int) {
	centerX := width / 2

	// Calculate bar lengths
	stdDevBarLen := int(math.Min(float64(item.stdDev*10), 100))
	scoreBarLen := int(math.Max(0, 100-item.avgRank*10))

	// Render std dev value
	stdDevStr := fmt.Sprintf("%.1f", item.stdDev)
	barStartX := centerX - len(item.ID)/2 - 2
	stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
	if stdDevX < 0 {
		stdDevX = 0
	}
	r.writeString(stdDevX, row, stdDevStr, style)

	// Render std dev bar (growing left from center)
	for j := 0; j < stdDevBarLen && centerX-len(item.ID)/2-j-2 >= 0; j++ {
		r.screen.SetContent(centerX-len(item.ID)/2-j-2, row, '=', nil, style)
	}

	// Render item ID at center
	idX := centerX - len(item.ID)/2
	if idX < 0 {
		idX = 0
	}
	r.writeString(idX, row, item.ID, style)

	// Render score bar with item text
	startX := centerX + len(item.ID)/2 + 1
	availableWidth := scoreBarLen
	if startX+availableWidth > width {
		availableWidth = width - startX
	}

	if availableWidth > 0 {
		displayText := item.Value
		if len(displayText) > availableWidth {
			displayText = displayText[:availableWidth]
		}

		// Render the text
		for i, ch := range displayText {
			if startX+i < width {
				r.screen.SetContent(startX+i, row, ch, nil, style)
			}
		}

		// Fill remaining space with → characters
		for j := len(displayText); j < availableWidth && startX+j < width; j++ {
			r.screen.SetContent(startX+j, row, '=', nil, style)
		}
	}

	// Render rank value
	rankStr := fmt.Sprintf("%.1f", item.avgRank)
	rankX := startX + scoreBarLen + 1
	if rankX+len(rankStr) > width {
		rankX = width - len(rankStr)
	}
	if rankX >= 0 {
		r.writeString(rankX, row, rankStr, style)
	}
}

func (r *Ranker) renderEliminatedItem(item *ItemDisplayState, stats *itemStats, row int, style tcell.Style, width int) {
	centerX := width / 2


	// Calculate bar lengths (smaller for eliminated items)
	stdDevBarLen := int(math.Min(float64(stats.stdDev*5), 50)) // Half size
	scoreBarLen := int(math.Max(0, 50-stats.avgRank*5))       // Half size

	// Render std dev value
	stdDevStr := fmt.Sprintf("%.1f", stats.stdDev)
	barStartX := centerX - len(item.ID)/2 - 2
	stdDevX := barStartX - stdDevBarLen - len(stdDevStr) - 1
	if stdDevX < 0 {
		stdDevX = 0
	}
	r.writeString(stdDevX, row, stdDevStr, style)

	// Render std dev bar (growing left from center)
	for j := 0; j < stdDevBarLen && centerX-len(item.ID)/2-j-2 >= stdDevX+len(stdDevStr)+1; j++ {
		r.screen.SetContent(centerX-len(item.ID)/2-j-2, row, '=', nil, style)
	}

	// Render item ID at center
	idX := centerX - len(item.ID)/2
	if idX < 0 {
		idX = 0
	}
	r.writeString(idX, row, item.ID, style)

	// Render score bar with abbreviated item text
	startX := centerX + len(item.ID)/2 + 1
	availableWidth := scoreBarLen
	if startX+availableWidth > width {
		availableWidth = width - startX
	}

	if availableWidth > 0 {
		displayText := item.Value
		if len(displayText) > availableWidth {
			displayText = displayText[:availableWidth]
		}

		// Render the text
		for i, ch := range displayText {
			if startX+i < width {
				r.screen.SetContent(startX+i, row, ch, nil, style)
			}
		}

		// Fill remaining space with → characters
		for j := len(displayText); j < availableWidth && startX+j < width; j++ {
			r.screen.SetContent(startX+j, row, '=', nil, style)
		}
	}

	// Render rank value
	rankStr := fmt.Sprintf("%.1f", stats.avgRank)
	rankX := startX + scoreBarLen + 1
	if rankX+len(rankStr) > width {
		rankX = width - len(rankStr)
	}
	if rankX >= 0 {
		r.writeString(rankX, row, rankStr, style)
	}
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

const promptFmt = "id: `%s`\nvalue:\n```\n%s\n```\n\n"

var promptDisclaimer = "\n\nREMEMBER to:\n" +
	"- ALWAYS respond with the short 6-8 character ID of each item found above the value " +
	"(i.e., I'll provide you with `id: <ID>` above the value, and you should respond with that same ID in your response)\n" +
	"— NEVER respond with the actual value!\n" +
	"— NEVER include backticks around IDs in your response!\n" +
	"— NEVER include scores or a written reason/justification in your response!\n" +
	"- Respond in RANKED DESCENDING order, where the FIRST item in your response is the MOST RELEVANT\n" +
	"- Respond in JSON format, with the following schema:\n  {\"objects\": [\"<ID1>\", \"<ID2>\", ...]}\n\n" +
	"Here are the objects to be ranked:\n\n"

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

	// Use tiktoken for OpenAI endpoints, simple approximation for custom endpoints
	if r.cfg.OpenAIAPIURL == "" {
		return len(r.encoding.Encode(text, nil, nil))
	} else {
		return len(text) / 4
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

		prompt := r.cfg.InitialPrompt + promptDisclaimer
		inputIDs := make(map[string]bool)

		if useMemorableIDs {
			// Use memorable IDs in the prompt
			for _, obj := range group {
				tempID := originalToTemp[obj.ID]
				prompt += fmt.Sprintf(promptFmt, tempID, obj.Value)
				inputIDs[tempID] = true
			}
		} else {
			// Fall back to original IDs
			for _, obj := range group {
				prompt += fmt.Sprintf(promptFmt, obj.ID, obj.Value)
				inputIDs[obj.ID] = true
			}
		}

		// Add reasoning request if enabled and not in threshing round
		if r.cfg.Reasoning && r.round > 1 {
			r.cfg.Logger.Debug("Adding reasoning request to prompt", "round", r.round, "batch", batchNumber)
			prompt += "\n\nIMPORTANT: In addition to ranking, you must also provide reasoning. For each item, write a brief 1-2 sentence explanation of why it ranked in that position relative to the others in this batch. Focus on distinctive features that made it rank higher or lower.\n\n"
			prompt += "Your response must include both:\n"
			prompt += "1. An 'objects' array with the ranked IDs\n"
			prompt += "2. A 'reasoning' object mapping each ID to its explanation\n\n"
			prompt += "Example format:\n"
			prompt += "{\n"
			prompt += "  \"objects\": [\"id1\", \"id2\", \"id3\"],\n"
			prompt += "  \"reasoning\": {\n"
			prompt += "    \"id1\": \"This item ranked highest because...\",\n"
			prompt += "    \"id2\": \"This item ranked second because...\",\n"
			prompt += "    \"id3\": \"This item ranked lowest because...\"\n"
			prompt += "  }\n"
			prompt += "}\n"
		}

		var rankedResponse rankedObjectResponse
		rankedResponse, err = r.callOpenAI(prompt, runNumber, batchNumber, inputIDs)
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
		if r.cfg.Reasoning && r.round > 1 && len(rankedResponse.Reasoning) > 0 {
			r.cfg.Logger.Debug("Received reasoning from API", "round", r.round, "batch", batchNumber, "count", len(rankedResponse.Reasoning))
		} else if r.cfg.Reasoning && r.round > 1 {
			r.cfg.Logger.Warn("No reasoning received from API despite requesting it", "round", r.round, "batch", batchNumber)
		}
		for i, id := range rankedResponse.Objects {
			for _, obj := range group {
				if obj.ID == id {
					reasoning := ""
					if rankedResponse.Reasoning != nil {
						reasoning = rankedResponse.Reasoning[id]
						if r.cfg.Reasoning && r.round > 1 {
							r.cfg.Logger.Debug("Attaching reasoning to rankedObject", "id", id, "hasReasoning", reasoning != "", "reasoningLen", len(reasoning))
						}
					}
					rankedObjects = append(rankedObjects, rankedObject{
						Object:    obj,
						Score:     float64(i + 1), // Score based on position (1 for first, 2 for second, etc.)
						Reasoning: reasoning,
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

		// Use the standard schema (reasoning field is always included and required)
		schema := rankedObjectResponseSchema

		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Messages: conversationHistory,
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "ranked_object_response",
						Description: openai.String("List of ranked object IDs"),
						Schema:      schema,
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
				conversationHistory = append(conversationHistory,
					openai.UserMessage(invalidJSONStr),
				)
				trimmedContent := strings.TrimSpace(completion.Choices[0].Message.Content)
				r.cfg.Logger.Debug("OpenAI API response", "content", trimmedContent)
				continue
			}

			missingIDs, err := validateIDs(&rankedResponse, inputIDs)
			if err != nil {
				r.logFromApiCall(runNum, batchNum, fmt.Sprintf("Missing IDs: [%s]", strings.Join(missingIDs, ", ")))
				conversationHistory = append(conversationHistory,
					openai.UserMessage(fmt.Sprintf(missingIDsStr, strings.Join(missingIDs, ", "))),
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
			return rankedObjectResponse{}, fmt.Errorf("iteration %d, batch %*d/%d: unexpected error: %w", runNum, len(strconv.Itoa(r.numBatches)), batchNum, r.numBatches, err)
		}
	}
}

// summarizeReasoning creates a 3-5 sentence summary from all reasoning snippets for an item
func (r *Ranker) summarizeReasoning(itemID string, itemValue string, snippets []string) (*ReasoningProsCons, error) {
	if len(snippets) == 0 {
		return nil, nil
	}

	// Skip summarization in dry run mode
	if r.cfg.DryRun {
		return nil, nil
	}

	prompt := fmt.Sprintf(`Below are %d reasoning snippets from different comparisons of the item "%s". These snippets come from various rounds where this item was compared against different sets of items.

Your task: Analyze these snippets and extract two things:
1. PROS: Points that generally weighed in favor of this item (2-4 sentences)
2. CONS: Points that generally weighed against this item (2-4 sentences)

Either pros or cons can be empty if there's nothing to say. For top-performing items, cons might be empty. For lower-performing items, pros might be minimal.

Critical style rules:
- DO NOT start sentences with "The item emphasizes", "The item focuses on", "This item", "It emphasizes", "It focuses", "It highlights"
- DO NOT end with "Overall," or "Overall, it" - just stop when you're done
- DO NOT use "While it", "Although it", "However," at the start of every other sentence
- DO NOT use formal academic language - write like you're taking quick notes
- DO NOT use absolute position language like "ranked highest", "ranked lowest", "ranked second"
- DO vary sentence structure - mix short and longer sentences, start sentences differently
- Just capture what's notable: what does it reference? what concepts appear? what stands out?

Write naturally and directly. Imagine explaining to a colleague in conversation.

Reasoning snippets:
`, len(snippets), itemValue)

	for i, snippet := range snippets {
		prompt += fmt.Sprintf("%d. %s\n", i+1, snippet)
	}

	prompt += "\nProvide your response in JSON format with two keys: 'pros' and 'cons'. Each value should be a string (or empty string if nothing to say).\n"
	prompt += `Example: {"pros": "Strong connection to X. References Y explicitly.", "cons": "Lacks specificity compared to items with Z."}`

	// Create a simple text completion request using OpenAI client
	clientOptions := []option.RequestOption{
		option.WithAPIKey(r.cfg.OpenAIKey),
		option.WithMaxRetries(3),
	}

	// Add base URL option if specified
	if r.cfg.OpenAIAPIURL != "" {
		baseURL := r.cfg.OpenAIAPIURL
		if !strings.HasSuffix(baseURL, "/") {
			baseURL += "/"
		}
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client := openai.NewClient(clientOptions...)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Generate JSON schema for strict mode
	schema := generateSchema[ReasoningProsCons]()

	completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		Model: r.cfg.OpenAIModel,
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:        "reasoning_pros_cons",
					Description: openai.String("Reasoning pros and cons for a ranked item"),
					Schema:      schema,
					Strict:      openai.Bool(true),
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("error calling OpenAI for summarization: %w", err)
	}

	if len(completion.Choices) == 0 || completion.Choices[0].Message.Content == "" {
		return nil, fmt.Errorf("empty response from OpenAI for summarization")
	}

	// Parse JSON response (no need to strip markdown fences with strict schema mode)
	content := completion.Choices[0].Message.Content

	var result ReasoningProsCons
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return nil, fmt.Errorf("error parsing summarization JSON: %w", err)
	}

	return &result, nil
}

