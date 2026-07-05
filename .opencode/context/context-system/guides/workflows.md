# Context Management Workflows

## Default: QuickScan + Harvest
```bash
/context
# Scans for summary files, recommends /context harvest
```

## Extract from Architecture Doc
```bash
/context extract from ARCHITECTURE.md
# Reads ARCHITECTURE.md, creates domain/concepts/*, lookup/*, guides/*
```

## Organize Existing Files
```bash
/context organize development/          # Move flat files into subdirectories
/context organize development/ --dry-run  # Preview only
```

## Update for Changes
```bash
/context update for "ML Inference API added"
# Adds new service to services.md, ports.md, updates navigation.md
```

## Add Recurring Error
```bash
/context error for "RabbitMQ connection refused"
# Creates errors/rabbitmq-connection.md with cause + fix
```
