package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	MatchesParsed = promauto.NewCounter(prometheus.CounterOpts{
		Name: "parser_matches_parsed_total",
		Help: "Total number of matches successfully parsed and stored",
	})

	MatchesFailed = promauto.NewCounter(prometheus.CounterOpts{
		Name: "parser_matches_failed_total",
		Help: "Total number of matches that failed to parse or store",
	})

	BatchProcessingDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "parser_batch_processing_duration_seconds",
		Help:    "Duration of batch processing operations",
		Buckets: prometheus.DefBuckets,
	})

	BatchSize = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "parser_batch_size",
		Help:    "Number of matches in each processed batch",
		Buckets: []float64{1, 5, 10, 25, 50, 100, 200},
	})

	DLQMessages = promauto.NewCounter(prometheus.CounterOpts{
		Name: "parser_dlq_messages_total",
		Help: "Total number of messages sent to DLQ",
	})
)
