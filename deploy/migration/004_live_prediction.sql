-- Live Match Prediction: model metadata table
-- Stores per-patch LiveDraftBERT model info for API loading.

CREATE TABLE IF NOT EXISTS live_prediction_models (
    patch_id            INT PRIMARY KEY,
    model_filename      TEXT NOT NULL,       -- draftbert_live_compiled_{patch}.pt
    weights_filename    TEXT NOT NULL,       -- draftbert_live_weights_{patch}.pt
    feature_columns     JSONB NOT NULL,      -- dynamic feature column list
    val_auc             FLOAT,
    val_logloss         FLOAT,
    n_matches           INT,
    n_samples           INT,
    trained_at          TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO _migrations (name) VALUES ('004_live_prediction.sql') ON CONFLICT DO NOTHING;
