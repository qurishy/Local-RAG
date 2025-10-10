namespace Local_Rag_Web.Models
{
    /// <summary>
    /// Result of a vector similarity search.
    /// Contains the chunk and its similarity score.
    /// </summary>
    public class SearchResult
    {
        public DocumentChunk Chunk { get; set; }
        public Document Document { get; set; }
        public double SimilarityScore { get; set; }
        public int ChunkIndex { get; set; }
    }
}
