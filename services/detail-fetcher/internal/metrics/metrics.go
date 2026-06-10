package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	MessagesReceivedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "detail_fetcher_messages_received_total",
		Help: "Total messages received from match_ids queue.",
	})

	FetchesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "detail_fetcher_fetches_total",
		Help: "Total OpenDota fetch attempts by result.",
	}, []string{"result"}) // "success", "not_found", "error"

	PublishesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "detail_fetcher_publishes_total",
		Help: "Total publishes to raw_matches queue by result.",
	}, []string{"result"}) // "success", "error"

	DLQRoutedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "detail_fetcher_dlq_routed_total",
		Help: "Total messages sent to DLQ after exhausted retries.",
	})

	FetchDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "detail_fetcher_fetch_duration_seconds",
		Help:    "Duration of OpenDota match fetch requests through proxy.",
		Buckets: []float64{.5, 1, 2, 5, 10, 30, 60},
	})

	PublishDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "detail_fetcher_publish_duration_seconds",
		Help:    "Duration of RabbitMQ publish operations.",
		Buckets: prometheus.DefBuckets,
	})
)
