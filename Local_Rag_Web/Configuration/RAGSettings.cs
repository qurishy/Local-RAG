namespace Local_Rag_Web.Configuration
{
    /// <summary>
    /// Configuration settings for the RAG system.
    /// These are loaded from appsettings.json and injected throughout the application.
    /// </summary>
    public class RAGSettings
    {
        // Path to the folder containing documents to index
        public string SharedFolderPath { get; set; }

        // Path to the embedding model file (ONNX format)
        public string EmbeddingModelPath { get; set; }

        // Path to the LLM model file (ONNX format)
        public string LLMModelPath { get; set; }

        // Dimension of the embedding vectors (e.g., 384 for all-MiniLM-L6-v2)
        public int EmbeddingDimension { get; set; } = 384;

        // Size of each document chunk in characters
        public int ChunkSize { get; set; } = 1000;

        // Overlap between chunks to maintain context
        public int ChunkOverlap { get; set; } = 200;

        // Number of top results to retrieve during search
        public int TopKResults { get; set; } = 5;

        // Similarity threshold (0-1) for considering a result relevant
        public double SimilarityThreshold { get; set; } = 0.7;

        // Maximum tokens for LLM response generation
        public int MaxResponseTokens { get; set; } = 512;

        // File extensions to process
        public string[] SupportedFileTypes { get; set; } =
            new[] { ".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx" };
    }
}
