package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	PaginationRunsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "id_fetcher_pagination_runs_total",
		Help: "Total fetch runs by outcome.",
	}, []string{"result"}) // "success" | "error" | "cancelled"

	MatchIDsPublishedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "id_fetcher_match_ids_published_total",
		Help: "Total match IDs published to RabbitMQ.",
	})

	APICallsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "id_fetcher_api_calls_total",
		Help: "Total OpenDota API calls by result.",
	}, []string{"status"}) // "ok" | "error"
)
