module github.com/dota-stratz/services/detail-fetcher

go 1.26.3

require (
	github.com/dota-stratz/shared/go-common v0.0.0-00010101000000-000000000000
	github.com/prometheus/client_golang v1.23.2
	github.com/rabbitmq/amqp091-go v1.11.0
	github.com/stretchr/testify v1.11.1
	go.uber.org/zap v1.28.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.16.1 // indirect
	github.com/redis/go-redis/v9 v9.20.0 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	go.uber.org/multierr v1.10.0 // indirect
	go.yaml.in/yaml/v2 v2.4.2 // indirect
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	google.golang.org/protobuf v1.36.8 // indirect
)

replace github.com/dota-stratz/shared/go-common => ../../shared/go-common
