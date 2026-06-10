group "default" {
  targets = ["id-fetcher", "detail-fetcher", "parser", "proxy-manager"]
}

target "id-fetcher" {
  context = "../"
  dockerfile = "services/id-fetcher/Dockerfile"
  # Image name must match what compose.yaml expects by default
  # (project=deploy + service=id-fetcher => deploy-id-fetcher:latest).
  tags = ["deploy-id-fetcher:latest"]
}

target "detail-fetcher" {
  context = "../"
  dockerfile = "services/detail-fetcher/Dockerfile"
  tags = ["deploy-detail-fetcher:latest"]
}

target "parser" {
  context = "../"
  dockerfile = "services/parser/Dockerfile"
  tags = ["deploy-parser:latest"]
}

target "proxy-manager" {
  context = "../"
  dockerfile = "services/proxy-manager/Dockerfile"
  tags = ["deploy-proxy-manager:latest"]
}

