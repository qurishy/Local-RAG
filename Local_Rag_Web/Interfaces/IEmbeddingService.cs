namespace Local_Rag_Web.Interfaces
{
    /// <summary>
    /// Service for generating embeddings from text.
    /// Embeddings are vector representations that capture semantic meaning.
    /// </summary>
    public interface IEmbeddingService
    {
        /// <summary>
        /// Generates an embedding vector for the given text.
        /// This converts human-readable text into a mathematical representation
        /// that can be used for similarity calculations.
        /// </summary>
        /// <param name="text">Input text to embed</param>
        /// <returns>Float array representing the embedding vector</returns>
        Task<float[]> GenerateEmbeddingAsync(string text);

        /// <summary>
        /// Generates embeddings for multiple texts in batch.
        /// More efficient than calling GenerateEmbeddingAsync multiple times.
        /// </summary>
        Task<List<float[]>> GenerateEmbeddingsAsync(List<string> texts);

        /// <summary>
        /// Gets the dimension of the embedding vectors produced by this service.
        /// </summary>
        int EmbeddingDimension { get; }
    }
}
