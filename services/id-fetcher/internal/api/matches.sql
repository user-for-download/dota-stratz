SELECT match_id, start_time
FROM matches
WHERE start_time >= (EXTRACT(EPOCH FROM NOW() - INTERVAL '%d days'))::BIGINT
AND lobby_type IN (%s)
ORDER BY match_id DESC
LIMIT 50000
