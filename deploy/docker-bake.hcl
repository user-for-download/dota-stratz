# Bake produces images tagged as deploy-<service>:latest.  Compose
# services reference these exact tags via `image:` + `pull_policy: never`
# so they find locally-baked images instead of trying to pull.

group "default" {
  targets = ["id-fetcher", "detail-fetcher", "parser", "proxy-manager"]
}

target "id-fetcher" {
  context = "../"
  dockerfile = "services/id-fetcher/Dockerfile"
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

target "trainer" {
  context = "../"
  dockerfile = "services/trainer/Dockerfile"
  tags = ["deploy-trainer:latest"]
}

target "api" {
  context = "../"
  dockerfile = "services/api/Dockerfile"
  tags = ["deploy-api:latest"]
}

target "frontend" {
  context = "../"
  dockerfile = "services/frontend/Dockerfile"
  tags = ["deploy-frontend:latest"]
}

