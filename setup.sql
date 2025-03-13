-- Enable the pgvector extension for vector-based search
create extension IF not exists vector;

-- Table to store knowledge chunks with metadata and embeddings
create table knowledge_base (
  id BIGSERIAL primary key, -- Unique identifier
  url VARCHAR not null, -- Source URL
  chunk_number INTEGER not null, -- Order of the chunk in the document
  title VARCHAR not null, -- Extracted title
  summary VARCHAR not null, -- Extracted summary
  content TEXT not null, -- Full chunk content
  metadata JSONB not null default '{}'::JSONB, -- Additional metadata
  embedding VECTOR (768), -- Gemini embedding (768 dimensions)
  created_at TIMESTAMPTZ default timezone ('utc', now()) not null, -- Timestamp in UTC
  unique (url, chunk_number) -- Prevent duplicate chunks for the same document
);

-- Index for efficient vector similarity search
create index idx_embedding on knowledge_base using ivfflat (embedding vector_cosine_ops);

-- Index for faster metadata filtering
create index idx_metadata on knowledge_base using GIN (metadata);

-- Function to search for similar knowledge chunks
create function match_knowledge_base (
  query_embedding VECTOR (768), -- Input embedding for similarity search
  match_count INT default 10, -- Number of results to return
  filter JSONB default '{}'::JSONB -- Optional metadata filter
) RETURNS table (
  id BIGINT,
  url VARCHAR,
  chunk_number INTEGER,
  title VARCHAR,
  summary VARCHAR,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
) LANGUAGE plpgsql as $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        id,
        url,
        chunk_number,
        title,
        summary,
        content,
        metadata,
        1 - (embedding <=> query_embedding) AS similarity  -- Cosine similarity calculation
    FROM knowledge_base
    WHERE metadata @> filter  -- Apply metadata filter if provided
    ORDER BY embedding <=> query_embedding  -- Sort by similarity
    LIMIT match_count;
END;
$$;

-- Enable Row-Level Security (RLS) for controlled access
alter table knowledge_base ENABLE row LEVEL SECURITY;

-- Allow public read access to the knowledge_base table
create policy allow_public_read on knowledge_base for
select
  to PUBLIC using (true);
