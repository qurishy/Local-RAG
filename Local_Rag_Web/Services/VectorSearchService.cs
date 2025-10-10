using Local_Rag_Web.Configuration;
using Local_Rag_Web.DATA;
using Local_Rag_Web.Interfaces;
using Local_Rag_Web.Models;
using Microsoft.Extensions.Options;
using Microsoft.EntityFrameworkCore;

namespace Local_Rag_Web.Services
{ /// <summary>
  /// Service for performing semantic similarity searches in the vector database.
  /// 
  /// Here's how vector search works conceptually:
  /// Imagine you have a library where instead of organizing books by title or author,
  /// you organize them by their meaning. Books about similar topics would be physically
  /// close to each other. When you search for something, you stand in the location that
  /// represents your query, and the closest books around you are the most relevant.
  /// 
  /// In our digital case, we use mathematical similarity (cosine similarity) to find
  /// the "closest" chunks of text to the user's query in this semantic space.
  /// </summary>
    public class VectorSearchService : IVectorSearchService
    {
        private readonly RAGDbContext _context;
        private readonly RAGSettings _settings;

        public VectorSearchService(RAGDbContext context, IOptions<RAGSettings> settings)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _settings = settings.Value;
        }

        public async Task<List<SearchResult>> SearchAsync(
            float[] queryEmbedding,
            int topK,
            double similarityThreshold)
        {
            if (queryEmbedding == null || queryEmbedding.Length == 0)
                throw new ArgumentException("Query embedding cannot be null or empty", nameof(queryEmbedding));

            // Step 1: Retrieve all document chunks with their embeddings
            // In a production system with millions of documents, you would want
            // to use a specialized vector database like Milvus, Pinecone, or Qdrant
            // that can perform approximate nearest neighbor search efficiently.
            // 
            // For a defense company with thousands of documents on a closed network,
            // this in-memory approach works well and keeps everything under your control.
            var allChunks = await _context.DocumentChunks
                .Include(c => c.Document)
                .Where(c => c.EmbeddingVector != null)
                .ToListAsync();

            // Step 2: Calculate similarity scores for each chunk
            var searchResults = new List<SearchResult>();

            foreach (var chunk in allChunks)
            {
                // Convert the stored byte array back to a float array
                var chunkEmbedding = DeserializeEmbedding(chunk.EmbeddingVector);

                // Calculate how similar this chunk is to the query
                // Cosine similarity ranges from -1 to 1, where:
                // - 1 means the vectors point in exactly the same direction (very similar)
                // - 0 means they're perpendicular (unrelated)
                // - -1 means they point in opposite directions (opposite meaning)
                double similarity = CalculateCosineSimilarity(queryEmbedding, chunkEmbedding);

                // Only include results that meet the similarity threshold
                // This filters out irrelevant results
                if (similarity >= similarityThreshold)
                {
                    searchResults.Add(new SearchResult
                    {
                        Chunk = chunk,
                        Document = chunk.Document,
                        SimilarityScore = similarity,
                        ChunkIndex = chunk.ChunkIndex
                    });
                }
            }

            // Step 3: Sort by similarity (highest first) and take top K results
            // The top K results are the most semantically similar to the query
            return searchResults
                .OrderByDescending(r => r.SimilarityScore)
                .Take(topK)
                .ToList();
        }

        /// <summary>
        /// Calculates cosine similarity between two vectors.
        /// 
        /// Think of cosine similarity like measuring the angle between two arrows
        /// in space. If they point in the same direction, the angle is 0° and
        /// similarity is 1. If they point in completely different directions,
        /// similarity is 0 or negative.
        /// 
        /// The formula is: similarity = (A · B) / (||A|| × ||B||)
        /// Where:
        /// - A · B is the dot product (sum of element-wise multiplication)
        /// - ||A|| and ||B|| are the magnitudes (lengths) of the vectors
        /// 
        /// Since we normalize our embeddings to unit length, ||A|| and ||B|| are both 1,
        /// so the formula simplifies to just the dot product.
        /// </summary>
        public double CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new ArgumentException("Vectors must have the same dimension");

            // Calculate dot product
            // This is the sum of multiplying corresponding elements
            double dotProduct = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
            }

            // If vectors are normalized (which ours are), the dot product
            // IS the cosine similarity. Otherwise, we'd need to divide by
            // the product of the vector magnitudes.
            return dotProduct;
        }

        /// <summary>
        /// Converts the byte array stored in the database back to a float array.
        /// We store embeddings as bytes to save database space and improve performance.
        /// </summary>
        private float[] DeserializeEmbedding(byte[] bytes)
        {
            if (bytes == null || bytes.Length == 0)
                return Array.Empty<float>();

            // Each float is 4 bytes, so the number of floats is bytes.Length / 4
            var floatCount = bytes.Length / sizeof(float);
            var floats = new float[floatCount];

            // Convert bytes back to floats
            // Buffer.BlockCopy is very fast for this operation
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);

            return floats;
        }

        /// <summary>
        /// Converts a float array to a byte array for database storage.
        /// This is the inverse of DeserializeEmbedding.
        /// </summary>
        public static byte[] SerializeEmbedding(float[] embedding)
        {
            if (embedding == null || embedding.Length == 0)
                return Array.Empty<byte>();

            // Allocate byte array: each float needs 4 bytes
            var bytes = new byte[embedding.Length * sizeof(float)];

            // Copy float data to byte array
            Buffer.BlockCopy(embedding, 0, bytes, 0, bytes.Length);

            return bytes;
        }
    }

    /// <summary>
    /// Extension methods for working with vectors.
    /// These utility methods help with common vector operations.
    /// </summary>
    public static class VectorExtensions
    {
        /// <summary>
        /// Calculates the Euclidean distance between two vectors.
        /// This is an alternative to cosine similarity - it measures
        /// the straight-line distance between two points in space.
        /// 
        /// Euclidean distance is useful when the magnitude of vectors matters,
        /// while cosine similarity only cares about direction.
        /// </summary>
        public static double EuclideanDistance(this float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new ArgumentException("Vectors must have the same dimension");

            double sumSquaredDifferences = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                var difference = vector1[i] - vector2[i];
                sumSquaredDifferences += difference * difference;
            }

            return Math.Sqrt(sumSquaredDifferences);
        }

        /// <summary>
        /// Calculates the magnitude (length) of a vector.
        /// This is the distance from the origin to the point represented by the vector.
        /// </summary>
        public static double Magnitude(this float[] vector)
        {
            double sumOfSquares = 0;
            foreach (var value in vector)
            {
                sumOfSquares += value * value;
            }

            return Math.Sqrt(sumOfSquares);
        }

        /// <summary>
        /// Normalizes a vector to unit length.
        /// After normalization, the vector has magnitude 1 but points
        /// in the same direction.
        /// </summary>
        public static float[] Normalize(this float[] vector)
        {
            var magnitude = vector.Magnitude();

            if (magnitude == 0)
                return vector;

            var normalized = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                normalized[i] = (float)(vector[i] / magnitude);
            }

            return normalized;
        }

        /// <summary>
        /// Calculates the dot product of two vectors.
        /// The dot product is the sum of element-wise multiplication.
        /// It's a fundamental operation in vector mathematics.
        /// </summary>
        public static double DotProduct(this float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new ArgumentException("Vectors must have the same dimension");

            double result = 0;
            for (int i = 0; i < vector1.Length; i++)
            {
                result += vector1[i] * vector2[i];
            }

            return result;
        }
    }
}
