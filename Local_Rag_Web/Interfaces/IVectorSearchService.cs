using Local_Rag_Web.Models;

namespace Local_Rag_Web.Interfaces
{ /// <summary>
  /// Service for performing vector similarity searches.
  /// This finds the most semantically similar documents to a query.
  /// </summary>
    public interface IVectorSearchService
    {
        /// <summary>
        /// Searches for the most similar document chunks to the query embedding.
        /// Uses cosine similarity to measure how close vectors are in semantic space.
        /// </summary>
        /// <param name="queryEmbedding">Embedding vector of the search query</param>
        /// <param name="topK">Number of top results to return</param>
        /// <param name="similarityThreshold">Minimum similarity score to include</param>
        /// <returns>List of search results sorted by similarity (highest first)</returns>
        Task<List<SearchResult>> SearchAsync(
            float[] queryEmbedding,
            int topK,
            double similarityThreshold);

        /// <summary>
        /// Calculates cosine similarity between two vectors.
        /// Returns a value between -1 and 1, where 1 means identical direction.
        /// </summary>
        double CalculateCosineSimilarity(float[] vector1, float[] vector2);
    }
}
