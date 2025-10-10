namespace Local_Rag_Web.Services
{ /// <summary>
  /// Service for splitting text into chunks with overlap.
  /// Chunking is critical for RAG systems because:
  /// 1. It allows embedding models to process manageable pieces
  /// 2. It enables more precise retrieval of relevant information
  /// 3. It keeps us within LLM token limits
  /// 
  /// The overlap between chunks ensures that information spanning
  /// chunk boundaries isn't lost.
  /// </summary>
    public class TextChunkingService
    {
        /// <summary>
        /// Splits text into overlapping chunks.
        /// </summary>
        /// <param name="text">The text to split</param>
        /// <param name="chunkSize">Target size of each chunk in characters</param>
        /// <param name="overlap">Number of characters to overlap between chunks</param>
        /// <returns>List of text chunks</returns>
        public List<string> ChunkText(string text, int chunkSize, int overlap)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new List<string>();

            var chunks = new List<string>();
            var position = 0;

            while (position < text.Length)
            {
                // Calculate the end position for this chunk
                var endPosition = Math.Min(position + chunkSize, text.Length);

                // Try to break at sentence boundaries for better readability
                // This is important for maintaining context
                if (endPosition < text.Length)
                {
                    // Look for sentence endings: period, exclamation, or question mark
                    var lastSentenceEnd = text.LastIndexOfAny(
                        new[] { '.', '!', '?' },
                        endPosition,
                        Math.Min(100, endPosition - position));

                    if (lastSentenceEnd > position)
                    {
                        endPosition = lastSentenceEnd + 1;
                    }
                }

                // Extract the chunk
                var chunk = text.Substring(position, endPosition - position).Trim();

                if (!string.IsNullOrWhiteSpace(chunk))
                {
                    chunks.Add(chunk);
                }

                // Move position forward, accounting for overlap
                // The overlap ensures context continuity between chunks
                position = endPosition - overlap;

                // Prevent infinite loops if overlap is too large
                if (position <= endPosition - chunkSize + overlap)
                {
                    position = endPosition;
                }
            }

            return chunks;
        }

        /// <summary>
        /// Estimates the number of tokens in text.
        /// This is a rough approximation: 1 token ≈ 4 characters for English.
        /// For more accurate counting, you'd use the specific tokenizer
        /// that matches your embedding model.
        /// </summary>
        public int EstimateTokenCount(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return 0;

            // Simple approximation: split by whitespace and punctuation
            return text.Split(new[] { ' ', '\n', '\r', '\t', ',', '.', ';', ':', '!', '?' },
                             StringSplitOptions.RemoveEmptyEntries).Length;
        }
    }
}
